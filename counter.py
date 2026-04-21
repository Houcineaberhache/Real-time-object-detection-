
import numpy as np
import pandas as pd
import supervision as sv
import time

from model_implementation import TARGET_CLASSES

class SignCounter:

    def __init__(self, lost_track_buffer=150,frames_to_confirm=4):
        self.tracker           = sv.ByteTrack(lost_track_buffer=lost_track_buffer)
        self.counted_ids       = set()
        self.total_count       = 0
        self.frames_to_confirm=frames_to_confirm
        self.detection_history = []
        self.first_seen_times  = {}
        self.current_frame_signs = []

        # counts how many consecutive frames each tracker_id has been seen
        # key = tracker_id, value = integer count
        self.consecutive_frames = {}

    def _to_sv_detections(self, houcine_detections):
        sign_dets = [d for d in houcine_detections if d["class_id"] in TARGET_CLASSES]
        if len(sign_dets) == 0:
            return sv.Detections.empty()
        xyxy       = np.array([list(d["bbox"])   for d in sign_dets], dtype=np.float32)
        confidence = np.array([d["confidence"]    for d in sign_dets], dtype=np.float32)
        class_id   = np.array([d["class_id"]      for d in sign_dets], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def process_frame(self, houcine_detections):
        self.current_frame_signs = []
        sv_detections = self._to_sv_detections(houcine_detections)
        tracked = self.tracker.update_with_detections(sv_detections)

        if len(tracked) == 0:
            return self._build_output()

        active_ids_this_frame = set()

        for i in range(len(tracked)):
            tracker_id = int(tracked.tracker_id[i])
            class_id   = int(tracked.class_id[i])
            label      = TARGET_CLASSES.get(class_id, "unknown")
            confidence = float(tracked.confidence[i])
            bbox       = tracked.xyxy[i].tolist()

            active_ids_this_frame.add(tracker_id)

            # increment consecutive frame counter for this id
            self.consecutive_frames[tracker_id] = self.consecutive_frames.get(tracker_id, 0) + 1

            # only count after seen for frames_to_confirm consecutive frames
            if tracker_id not in self.counted_ids:
                if self.consecutive_frames[tracker_id] >= self.frames_to_confirm:
                    self.counted_ids.add(tracker_id)
                    self.total_count += 1
                    self.first_seen_times[tracker_id] = time.time()
                    self._add_to_history(tracker_id, label, confidence, bbox)
                    print(f"[NEW]  id={tracker_id}  '{label}'  conf={confidence:.2f}  total={self.total_count}")
                    state = "NEW"
                else:
                    # seen but not confirmed yet — do not count, do not show
                    continue
            else:
                self._update_history(tracker_id, confidence)
                state = "CONTINUING"

            self.current_frame_signs.append({
                "tracker_id" : tracker_id,
                "label"      : label,
                "confidence" : round(confidence, 2),
                "state"      : state,
                "bbox"       : bbox
            })

        # reset consecutive counter for ids not seen this frame
        for tid in list(self.consecutive_frames.keys()):
            if tid not in active_ids_this_frame:
                self.consecutive_frames[tid] = 0

        return self._build_output()

    def _add_to_history(self, tracker_id, label, confidence, bbox):
        self.detection_history.append({
            "tracker_id"  : tracker_id,
            "label"       : label,
            "confidence"  : round(confidence, 2),
            "first_seen"  : time.strftime('%H:%M:%S'),
            "last_seen"   : time.strftime('%H:%M:%S'),
            "duration_sec": 0.0,
            "bbox_first"  : [int(v) for v in bbox]
        })

    def _update_history(self, tracker_id, confidence):
        for record in self.detection_history:
            if record["tracker_id"] == tracker_id:
                record["last_seen"]    = time.strftime('%H:%M:%S')
                record["confidence"]   = round(confidence, 2)
                record["duration_sec"] = round(time.time() - self.first_seen_times[tracker_id], 2)
                break

    def _build_output(self):
        return {
            "total_count"       : self.total_count,
            "current_signs"     : self.current_frame_signs,
            "detection_history" : self.detection_history,
        }

    def show_history_table(self):
        if not self.detection_history:
            print("No traffic signs detected yet.")
            return
        df = pd.DataFrame(self.detection_history)
        df.columns = ["ID", "Label", "Confidence", "First Seen", "Last Seen", "Duration (s)", "First BBox"]
        df.index = df.index + 1
        display(df)
        print(f"\n✅ Total unique traffic signs detected: {self.total_count}")

    def save_history_to_csv(self, filename="detection_history.csv"):
        if not self.detection_history:
            print("Nothing to save yet.")
            return
        pd.DataFrame(self.detection_history).to_csv(filename, index=False)
        print(f"✅ Saved to '{filename}'")
