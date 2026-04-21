import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import cv2

class CentroidTracker:
    def __init__(self, max_disappeared=25, max_distance=150):
        self.next_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.bboxes[oid]
                    del self.disappeared[oid]
            return self.objects, self.bboxes

        new_centroids = np.array([d[0] for d in detections], dtype="int")
        new_bboxes = [d[1] for d in detections]

        if len(self.objects) == 0:
            for i in range(len(new_centroids)):
                self.register(new_centroids[i], new_bboxes[i])
        else:
            ids = list(self.objects.keys())
            old_cents = list(self.objects.values())
            D = dist.cdist(np.array(old_cents), new_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                oid = ids[row]
                self.objects[oid] = new_centroids[col]
                self.bboxes[oid] = new_bboxes[col]
                self.disappeared[oid] = 0
                used_rows.add(row)
                used_cols.add(col)

            for col in set(range(len(new_centroids))) - used_cols:
                self.register(new_centroids[col], new_bboxes[col])

            for row in set(range(len(ids))) - used_rows:
                oid = ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.bboxes[oid]
                    del self.disappeared[oid]

        return self.objects, self.bboxes


class PedestrianCrosswalkCounter:
    def __init__(self, min_travel_px=65):
        # Initialisation du tracker de Salah
        self.tracker = CentroidTracker(max_disappeared=90, max_distance=250)
        self.min_travel_px = min_travel_px
        self.roi_entry_x = {}
        self.counted_ids = set()
        self.crossing_count = 0
        
        # ROI Ultra-Précis : Focus uniquement sur la voie du véhicule (Ego-Lane)
        self.roi_polygon = np.array([
            [610, 440], [670, 440], [1260, 720], [20, 720]
        ], dtype=np.int32)

    def feet_in_polygon(self, bbox):
        x1, y1, x2, y2 = bbox
        feet_x = (x1 + x2) // 2
        feet_y = y2
        return cv2.pointPolygonTest(self.roi_polygon, (float(feet_x), float(feet_y)), False) >= 0

    def process_frame(self, detections_raw):
        in_roi_pedestrians = []
        all_cars = []
        
        # 1. Filter and identify pedestrians vs cars
        for d in detections_raw:
            class_name = d["class_name"]
            bbox = d["bbox"]
            if class_name == "person":
                if self.feet_in_polygon(bbox):
                    in_roi_pedestrians.append(d)
            elif class_name in ["car", "truck"]:
                all_cars.append(d)

        # 2. Track objects in ROI
        track_input = [(((d["bbox"][0] + d["bbox"][2]) // 2, (d["bbox"][1] + d["bbox"][3]) // 2), d["bbox"]) for d in in_roi_pedestrians]
        tracked_objects, tracked_bboxes = self.tracker.update(track_input)
        
        current_frame_data = []

        for obj_id, (cx, cy) in tracked_objects.items():
            x1, y1, x2, y2 = tracked_bboxes[obj_id]
            feet_x = (x1 + x2) // 2
            
            # --- PRECISION FIX: Depth/Occlusion Logic ---
            # If a pedestrian is 'above' a car (smaller y2) and their x-ranges overlap,
            # it means the car is between the camera and the pedestrian.
            is_occluded = False
            for car in all_cars:
                cx1, cy1, cx2, cy2 = car["bbox"]
                
                # Check if car is closer (cy2 > y2)
                if cy2 > y2:
                    # Check horizontal overlap (IoU in X-axis)
                    overlap_x1 = max(x1, cx1)
                    overlap_x2 = min(x2, cx2)
                    if overlap_x2 > overlap_x1:
                        # Significant horizontal overlap
                        overlap_width = overlap_x2 - overlap_x1
                        ped_width = x2 - x1
                        if overlap_width / ped_width > 0.4: # 40% overlap
                            is_occluded = True
                            break

            if obj_id not in self.roi_entry_x:
                self.roi_entry_x[obj_id] = feet_x

            # Only count and alert if NOT occluded
            if not is_occluded:
                if obj_id not in self.counted_ids:
                    travel = abs(feet_x - self.roi_entry_x[obj_id])
                    if travel >= self.min_travel_px:
                        self.counted_ids.add(obj_id)
                        self.crossing_count += 1

            current_frame_data.append({
                "id": obj_id,
                "bbox": (x1, y1, x2, y2),
                "cx": cx, "cy": cy,
                "already_counted": obj_id in self.counted_ids,
                "is_occluded": is_occluded # Added for UI/debug
            })

        return {
            "crossing_count": self.crossing_count,
            "tracked_pedestrians": current_frame_data,
            "roi_polygon": self.roi_polygon
        }