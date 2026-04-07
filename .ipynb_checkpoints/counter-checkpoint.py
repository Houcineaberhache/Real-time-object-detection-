
# ════════════════════════════════════════════════════════════
# counter.py
# Member 3 — Abdo — Counting & Event Management
# INPUT  : detections list from Houcine's detect() each frame
# OUTPUT : dictionary that Mohammed reads
# ════════════════════════════════════════════════════════════


# ── SECTION 1: IMPORTS ───────────────────────────────────────
# numpy     → math library (needed to build arrays for ByteTrack)
# pandas    → turns your history list into a pretty table
# supervision → contains ByteTrack tracker
# time      → lets you record timestamps (when a sign was seen)

import numpy as np
import pandas as pd
import supervision as sv
import time


# ── SECTION 2: WHICH CLASSES COUNT AS TRAFFIC SIGNS ─────────
# These are the class_id numbers that Houcine's detect() uses
# We ONLY count these — persons, cars, trucks are ignored

TRAFFIC_SIGN_CLASSES = {
    9  : "traffic light",
    11 : "stop sign",
}


# ── SECTION 3: THE MAIN CLASS ────────────────────────────────
# A class is like a template. When you write SignCounter()
# you create one working copy of this template.

class SignCounter:

    # ── 3A: SETUP (runs once when you write SignCounter()) ───
    def __init__(self, lost_track_buffer=90):
        """
        lost_track_buffer : how many frames a sign can disappear
                            before we stop tracking it.
                            30 frames ≈ 1 second at 30fps.
        """

        # ByteTrack is the tracker — it gives each sign a
        # persistent ID that stays the same across frames.
        # Same physical sign = same tracker_id every frame.
        self.tracker = sv.ByteTrack(lost_track_buffer=lost_track_buffer)

        # A set is like a list but NEVER stores duplicates.
        # We store tracker_ids here the moment we first see them.
        # Before counting: ask "is this id already in counted_ids?"
        # If YES  → already counted, skip
        # If NO   → new sign, count it, then add to set
        self.counted_ids = set()

        # running total of unique signs counted so far
        self.total_count = 0

        # list of dictionaries — one record per unique sign
        # this is the "log" Mohammed will display
        self.detection_history = []

        # stores the exact second each tracker_id was first seen
        # key = tracker_id (number), value = time.time() (a float)
        self.first_seen_times = {}

        # signs visible in the current frame (reset every frame)
        self.current_frame_signs = []


    # ── 3B: CONVERT HOUCINE'S FORMAT → BYTETRACK FORMAT ─────
    def _to_sv_detections(self, houcine_detections):
        """
        Houcine's detect() returns a plain Python list of dicts.
        ByteTrack needs a special supervision Detections object.
        This function converts between the two formats.
        """

        # keep only traffic sign classes — drop persons, cars, trucks
        sign_dets = [
            d for d in houcine_detections
            if d["class_id"] in TRAFFIC_SIGN_CLASSES
        ]

        # if no traffic signs in this frame, return an empty object
        if len(sign_dets) == 0:
            return sv.Detections.empty()

        # ByteTrack needs numpy arrays (not plain Python lists)

        # bounding boxes — one row per sign — [x1, y1, x2, y2]
        xyxy = np.array(
            [list(d["bbox"]) for d in sign_dets],
            dtype=np.float32
        )

        # confidence scores — one number per sign
        confidence = np.array(
            [d["confidence"] for d in sign_dets],
            dtype=np.float32
        )

        # class ids — one number per sign (9 or 11)
        class_id = np.array(
            [d["class_id"] for d in sign_dets],
            dtype=int
        )

        # build and return the supervision Detections object
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )


    # ── 3C: PROCESS ONE FRAME — the main method ──────────────
    def process_frame(self, houcine_detections):
        """
        Houcine calls this once per frame.
        houcine_detections : the list returned by his detect()
        Returns            : output dict that Mohammed reads
        """

        # reset the visible signs list for this new frame
        self.current_frame_signs = []

        # Step 1 — convert Houcine's format to supervision format
        sv_detections = self._to_sv_detections(houcine_detections)

        # Step 2 — run ByteTrack
        # ByteTrack compares this frame's boxes to previous frames
        # and fills in tracker_id on each detection
        tracked = self.tracker.update_with_detections(sv_detections)

        # if ByteTrack returned nothing, return current output as-is
        if len(tracked) == 0:
            return self._build_output()

        # Step 3 — loop through each tracked sign this frame
        for i in range(len(tracked)):

            # ByteTrack's persistent ID for this sign
            tracker_id = int(tracked.tracker_id[i])

            # class number (9 or 11)
            class_id = int(tracked.class_id[i])

            # human-readable name
            label = TRAFFIC_SIGN_CLASSES.get(class_id, "unknown")

            # how confident YOLO is (0 to 1)
            confidence = float(tracked.confidence[i])

            # bounding box [x1, y1, x2, y2]
            bbox = tracked.xyxy[i].tolist()

            # ── NEW sign: first time we see this tracker_id ──
            if tracker_id not in self.counted_ids:

                # add to set so we never count this ID again
                self.counted_ids.add(tracker_id)

                # increment unique sign counter
                self.total_count += 1

                # record when it was first seen
                self.first_seen_times[tracker_id] = time.time()

                # add a new record to the history log
                self._add_to_history(tracker_id, label, confidence, bbox)

                # print so you can watch in real time
                print(f"[NEW]  id={tracker_id}  '{label}'  "
                      f"conf={confidence:.2f}  total={self.total_count}")

                state = "NEW"

            # ── CONTINUING sign: already counted before ──────
            else:
                # update last_seen time and duration in history
                self._update_history(tracker_id, confidence)
                state = "CONTINUING"

            # add this sign to current frame list
            self.current_frame_signs.append({
                "tracker_id" : tracker_id,
                "label"      : label,
                "confidence" : round(confidence, 2),
                "state"      : state,
                "bbox"       : bbox
            })

        # Step 4 — return the output package
        return self._build_output()


    # ── 3D: ADD A NEW RECORD TO HISTORY ──────────────────────
    def _add_to_history(self, tracker_id, label, confidence, bbox):

        record = {
            "tracker_id"  : tracker_id,
            "label"       : label,
            "confidence"  : round(confidence, 2),
            "first_seen"  : time.strftime('%H:%M:%S'),
            "last_seen"   : time.strftime('%H:%M:%S'),
            "duration_sec": 0.0,
            "bbox_first"  : [int(v) for v in bbox]
        }
        self.detection_history.append(record)


    # ── 3E: UPDATE AN EXISTING HISTORY RECORD ────────────────
    def _update_history(self, tracker_id, confidence):

        for record in self.detection_history:
            if record["tracker_id"] == tracker_id:
                record["last_seen"]    = time.strftime('%H:%M:%S')
                record["confidence"]   = round(confidence, 2)
                first_t = self.first_seen_times.get(tracker_id, time.time())
                record["duration_sec"] = round(time.time() - first_t, 2)
                break


    # ── 3F: BUILD THE OUTPUT DICT FOR MOHAMMED ───────────────
    def _build_output(self):
        return {
            "total_count"       : self.total_count,
            "current_signs"     : self.current_frame_signs,
            "detection_history" : self.detection_history,
        }


    # ── 3G: SHOW HISTORY AS A TABLE IN JUPYTER ───────────────
    def show_history_table(self):
        if not self.detection_history:
            print("No traffic signs detected yet.")
            return
        df = pd.DataFrame(self.detection_history)
        df.columns = [
            "ID", "Label", "Confidence",
            "First Seen", "Last Seen", "Duration (s)", "First BBox"
        ]
        df.index = df.index + 1
        display(df)
        print(f"\n✅ Total unique traffic signs detected: {self.total_count}")


    # ── 3H: SAVE HISTORY TO CSV FILE ─────────────────────────
    def save_history_to_csv(self, filename="detection_history.csv"):
        if not self.detection_history:
            print("Nothing to save yet.")
            return
        pd.DataFrame(self.detection_history).to_csv(filename, index=False)
        print(f"✅ History saved to '{filename}'")
