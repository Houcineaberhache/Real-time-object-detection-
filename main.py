import cv2
import numpy as np
import time
import threading
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from interaction_module import InteractionManager
from model_implementation import detect
from counter import SignCounter
from pedestrian_tracker import PedestrianCrosswalkCounter

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

global_stats = {
    "fps": 0,
    "persons_in_zone": 0,
    "stop_signs": 0,
    "traffic_lights": 0,
    "total_unique_signs": 0,
    "status": "running"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Danger-Zone geometry (pixels in 1280×720 frame)
#
#  We define a TRAPEZOID that covers ONLY the ego lane directly ahead.
#  Anything to the left (oncoming traffic) or sidewalks is EXCLUDED.
#
#   Top edge  : x ∈ [560, 720]  at  y = 440   (near horizon, narrow)
#   Bottom edge: x ∈ [180, 1100] at  y = 720   (bottom of frame, wide)
#
#  A detected object must have its CENTER-X (cx) fall inside this trapezoid's
#  horizontal span at its Y-level to count as "in our lane".
#
#  Helper: given y, return the allowed x range using linear interpolation.
# ─────────────────────────────────────────────────────────────────────────────
LANE_TOP_Y    = 440
LANE_BOT_Y    = 720
LANE_TOP_X1   = 540   # left edge at horizon
LANE_TOP_X2   = 740   # right edge at horizon
LANE_BOT_X1   = 150   # left edge at bottom
LANE_BOT_X2   = 1130  # right edge at bottom

def in_lane(cx, cy):
    """Returns True if (cx,cy) is inside the ego-lane trapezoid."""
    if cy < LANE_TOP_Y or cy > LANE_BOT_Y:
        return False
    t = (cy - LANE_TOP_Y) / (LANE_BOT_Y - LANE_TOP_Y)   # 0..1
    left  = LANE_TOP_X1 + t * (LANE_BOT_X1 - LANE_TOP_X1)
    right = LANE_TOP_X2 + t * (LANE_BOT_X2 - LANE_TOP_X2)
    return left <= cx <= right


class VideoProcessor:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_source}")

        self.video_fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_delay  = 1.0 / self.video_fps

        self.voice        = InteractionManager()
        self.sign_counter = SignCounter(lost_track_buffer=150, frames_to_confirm=4)
        self.ped_counter  = PedestrianCrosswalkCounter(min_travel_px=65)

        self.frame          = None
        self.processed_frame = None
        self.running        = True
        self.lock           = threading.Lock()

        # ── State machines ──────────────────────────────────────────────────
        # Vehicle danger: only warn ONCE per "encounter"
        self.danger_active      = False
        self.danger_clear_count = 0       # frames since danger zone was empty

        # Traffic light: only warn when color CHANGES, with hysteresis
        self.last_light_color       = None   # last announced color
        self.candidate_color        = None   # color we are currently seeing
        self.candidate_count        = 0      # consecutive frames we've seen it
        self.LIGHT_CONFIRM_FRAMES   = 2      # must see same color 2× before speaking
        self.light_absent_count     = 0      # frames with no light detected
        self.LIGHT_RESET_FRAMES     = 30     # frames of absence to reset memory

        # Pedestrian: only warn once until they leave the zone
        self.ped_in_zone_prev = False
        # Grace period: suppress false alerts for a few frames after video loop
        self.post_loop_grace  = 0

        # Start threads
        self.read_thread    = threading.Thread(target=self._read_frames,    daemon=True)
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.read_thread.start()
        self.process_thread.start()

    # ── Video reader ─────────────────────────────────────────────────────────
    def _read_frames(self):
        while self.running:
            try:
                t0 = time.time()
                ok, frame = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    # ── Hard reset: everything fires fresh on next loop ──────
                    self.danger_active          = False
                    self.danger_clear_count     = 0
                    self.last_light_color       = None
                    self.candidate_color        = None
                    self.candidate_count        = 0
                    self.light_absent_count     = 0
                    # Reset sign and pedestrian trackers completely
                    self.sign_counter = SignCounter(lost_track_buffer=150, frames_to_confirm=4)
                    self.ped_counter  = PedestrianCrosswalkCounter(min_travel_px=65)
                    # Reset voice state
                    self.voice.spoken_once.clear()
                    self.voice.cooldowns.clear()
                    self.voice.last_global_alert_time = 0
                    self.voice._speaking = False
                    self.post_loop_grace = 60  # suppress false-positives for first ~2s after loop
                    print("[VIDEO] Looped — FULL state reset.")
                    continue
                with self.lock:
                    self.frame = cv2.resize(frame, (1280, 720))
                sleep = max(0, self.frame_delay - (time.time() - t0))
                time.sleep(sleep)
            except Exception as e:
                print(f"[READER] {e}")
                time.sleep(0.5)

    # ── Main processing loop ─────────────────────────────────────────────────
    def _process_frames(self):
        global global_stats
        prev_time = time.time()

        while self.running:
            try:
                if self.frame is None:
                    time.sleep(0.01)
                    continue

                with self.lock:
                    frame_copy = self.frame.copy()

                # ── 1. YOLO Detection ────────────────────────────────────────
                detections, annotated = detect(frame_copy, conf_threshold=0.35)

                # Decrement post-loop grace period (suppresses false alerts after restart)
                if self.post_loop_grace > 0:
                    self.post_loop_grace -= 1

                # ── 2. Draw Ego-Lane overlay ──────────────────────────────────
                lane_poly = np.array([
                    [LANE_TOP_X1, LANE_TOP_Y], [LANE_TOP_X2, LANE_TOP_Y],
                    [LANE_BOT_X2, LANE_BOT_Y], [LANE_BOT_X1, LANE_BOT_Y]
                ], dtype=np.int32)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [lane_poly], (0, 60, 0))
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                cv2.polylines(annotated, [lane_poly], isClosed=True, color=(0, 200, 0), thickness=1)

                # ── 3. Vehicle Proximity (Danger Zone) ────────────────────────
                danger_this_frame = False
                light_detected_this_frame = False
                best_light_color = None  # will hold the color from the most prominent light

                for obj in detections:
                    label = obj["class_name"]
                    x1, y1, x2, y2 = obj["bbox"]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # ── Car / Truck: only if in ego-lane AND very close ───────
                    if label in ("car", "truck"):
                        if in_lane(cx, y2) and y2 > 620:  # higher threshold = only warn when truly close
                            danger_this_frame = True
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.putText(annotated, "DANGER: TOO CLOSE", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    # ── Traffic Lights: within the center horizontal band ─────
                    if label == "traffic light":
                        if 350 < cx < 950:  # tighter range, ignores far-side building lights
                            color = obj.get("light_color", "UNKNOWN")
                            if color != "UNKNOWN" and best_light_color is None:
                                best_light_color = color
                            light_detected_this_frame = True
                            # Draw light color label on frame
                            clr = (0, 0, 255) if color == "RED" else (0, 255, 0) if color == "GREEN" else (200, 200, 0)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), clr, 3)
                            cv2.putText(annotated, f"LIGHT: {color}", (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

                # ── Vehicle danger state machine ──────────────────────────────
                if danger_this_frame:
                    self.danger_clear_count = 0
                    if not self.danger_active:
                        # Don't warn if light is RED — we're stopped at an intersection
                        # along with the car ahead. Only warn if light is GREEN or unknown.
                        if self.last_light_color != "RED":
                            self.danger_active = True
                            self.voice.say("vehicle_ahead", "Vehicle ahead, keep your distance", priority=True)
                else:
                    self.danger_clear_count += 1
                    if self.danger_clear_count > 90:  # ~3 seconds of clear road before reset
                        if self.danger_active:
                            self.danger_active = False
                            self.voice.reset_once("vehicle_ahead")

                # ── Traffic light state machine (hysteresis) ──────────────────
                if light_detected_this_frame and best_light_color:
                    self.light_absent_count = 0
                    if best_light_color == self.candidate_color:
                        self.candidate_count += 1
                    else:
                        self.candidate_color = best_light_color
                        self.candidate_count = 1

                    # Only announce after LIGHT_CONFIRM_FRAMES stable frames
                    if self.candidate_count >= self.LIGHT_CONFIRM_FRAMES:
                        if self.candidate_color != self.last_light_color:
                            self.last_light_color = self.candidate_color
                            if self.candidate_color == "RED":
                                self.voice.say("light_red", "Traffic light is Red", once=False)
                            elif self.candidate_color == "GREEN":
                                self.voice.say("light_green", "Traffic light is Green, go", once=False)
                else:
                    self.light_absent_count += 1
                    if self.light_absent_count > self.LIGHT_RESET_FRAMES:
                        # Light is gone; reset so next intersection announces fresh
                        self.last_light_color  = None
                        self.candidate_color   = None
                        self.candidate_count   = 0
                        self.light_absent_count = 0

                # ── 4. Sign Counter ────────────────────────────────────────────
                sign_results = self.sign_counter.process_frame(detections)
                stop_count  = 0
                light_count = 0
                for obj in sign_results.get("current_signs", []):
                    lbl = obj["label"]
                    tid = obj["tracker_id"]
                    if lbl == "stop sign":     stop_count  += 1
                    if lbl == "traffic light": light_count += 1

                    # ONLY announce stop signs — not cars, not traffic lights (handled above)
                    if lbl == "stop sign":
                        x1, y1, x2, y2 = obj["bbox"]
                        area = (x2 - x1) * (y2 - y1)
                        if area > 900:
                            key = f"stop_{tid}"
                            self.voice.say(key, "Stop sign ahead, prepare to stop", once=True)

                # ── 5. Pedestrian Tracking ─────────────────────────────────────
                ped_results = self.ped_counter.process_frame(detections)
                roi_polygon = ped_results["roi_polygon"]

                ov2 = annotated.copy()
                cv2.fillPoly(ov2, [roi_polygon], (0, 0, 140))
                cv2.addWeighted(ov2, 0.18, annotated, 0.82, 0, annotated)
                cv2.polylines(annotated, [roi_polygon], isClosed=True, color=(0, 0, 255), thickness=1)

                ped_in_zone = 0
                for ped in ped_results["tracked_pedestrians"]:
                    px1, py1, px2, py2 = ped["bbox"]
                    is_occ = ped.get("is_occluded", False)
                    if is_occ:
                        col = (100, 100, 100)
                        cv2.putText(annotated, "BEHIND CAR", (int(px1), int(py1) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        ped_in_zone += 1
                        col = (0, 220, 0)
                        cv2.circle(annotated, (ped["cx"], ped["cy"]), 5, (0, 255, 255), -1)
                        # Only warn if pedestrian is CLOSE (feet in bottom 40% of frame)
                        # and past the grace period after a video loop
                        if not ped["already_counted"] and py2 > 430 and self.post_loop_grace == 0:
                            self.voice.say("person", "Watch out, pedestrian ahead")
                    cv2.rectangle(annotated, (int(px1), int(py1)), (int(px2), int(py2)), col, 2)

                # ── 6. FPS & Stats ────────────────────────────────────────────
                now  = time.time()
                fps  = 1.0 / max(now - prev_time, 0.001)
                prev_time = now

                global_stats.update({
                    "fps": round(fps, 1),
                    "persons_in_zone": ped_in_zone,
                    "stop_signs": stop_count,
                    "traffic_lights": light_count,
                    "total_unique_signs": sign_results.get("total_count", 0)
                })

                with self.lock:
                    self.processed_frame = annotated

            except Exception as e:
                print(f"[PROCESS] {e}")
                time.sleep(0.02)

    @staticmethod
    def _sign_phrase(label):
        phrases = {
            "stop sign":     "Stop sign ahead, prepare to stop",
            "traffic light": "Traffic light approaching",
        }
        return phrases.get(label, f"{label} detected")

    def get_frame_bytes(self):
        with self.lock:
            if self.processed_frame is None:
                return None
            ok, buf = cv2.imencode('.jpg', self.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return buf.tobytes() if ok else None


# ── FastAPI app ───────────────────────────────────────────────────────────────
video_path = os.path.join(BASE_DIR, 'video.mp4')
processor  = VideoProcessor(video_path)


async def generate_frames():
    while True:
        data = processor.get_frame_bytes()
        if data:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
        await asyncio.sleep(0.01)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/metrics")
async def get_metrics():
    return global_stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="warning")