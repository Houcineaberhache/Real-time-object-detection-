
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Houcine's class dictionary
TARGET_CLASSES = {
    0  : "person",
    9  : "traffic light",
    11 : "stop sign",
    2  : "car",
    7  : "truck",
}

# Houcine's color dictionary
CLASS_COLORS = {
    0  : (255, 200, 0),
    9  : (0,   255, 100),
    11 : (255, 50,  50),
    2  : (150, 150, 150),
    7  : (100, 100, 100),
}

# Load the YOLO model (downloads automatically if not present)
model = YOLO("yolov8n.pt")

# Houcine's detect function — copied exactly
def detect(frame, conf_threshold=0.45):
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    detections = []
    annotated = frame.copy()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in TARGET_CLASSES:
                continue
            confidence = float(box.conf[0])
            class_name = TARGET_CLASSES[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id"  : class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox"      : (x1, y1, x2, y2),
            })
            color     = CLASS_COLORS.get(class_id, (255, 255, 255))
            color_bgr = (color[2], color[1], color[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return detections, annotated

print("✓ Houcine's detect() loaded and ready")
