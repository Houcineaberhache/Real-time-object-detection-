from ultralytics import YOLO
import cv2
import numpy as np  # FIXED: was missing, caused 'np is not defined' error
import torch

model = YOLO("yolov8n.pt")  # downloads ~6MB automatically

TARGET_CLASSES = {
    0  : "person",
    9  : "traffic light",
    11 : "stop sign",
    2  : "car",
    7  : "truck",
}

CLASS_COLORS = {
    0  : (255, 200, 0),    # person       → yellow
    9  : (0,   255, 100),  # traffic light → green
    11 : (255, 50,  50),   # stop sign    → red
    2  : (150, 150, 150),  # car          → gray
    7  : (100, 100, 100),  # truck        → dark gray
}


def get_traffic_light_color(crop):
    """
    Analyzes a BGR crop of a traffic light and returns 'RED', 'GREEN', or 'UNKNOWN'.
    Uses HSV color space and checks top zone for red, bottom zone for green.
    """
    if crop is None or crop.size == 0:
        return "UNKNOWN"

    h, w = crop.shape[:2]
    if h < 6 or w < 3:
        return "UNKNOWN"  # crop too small to analyze

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # ── Red: appears at top of traffic light ─────────────────────────────────
    # Red hue wraps around 0°/180° in HSV
    m1 = cv2.inRange(hsv, np.array([0,   120,  80]), np.array([12,  255, 255]))
    m2 = cv2.inRange(hsv, np.array([158, 120,  80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    # ── Green: appears at bottom of traffic light ─────────────────────────────
    green_mask = cv2.inRange(hsv, np.array([45, 80, 80]), np.array([95, 255, 255]))

    # Analyse top 40% for red, bottom 40% for green
    top    = red_mask  [0 : int(h * 0.45), :]
    bottom = green_mask[int(h * 0.55) : h, :]

    red_px   = cv2.countNonZero(top)
    green_px = cv2.countNonZero(bottom)

    # Need at minimum 4 lit pixels to avoid noise
    if red_px   > green_px and red_px   >= 4: return "RED"
    if green_px > red_px   and green_px >= 4: return "GREEN"
    return "UNKNOWN"


def detect(frame, conf_threshold=0.35):
    """
    Run YOLOv8 on a single frame.

    Args:
        frame          : BGR image (numpy array from OpenCV)
        conf_threshold : minimum confidence to keep a detection

    Returns:
        detections : list of dicts  {class_id, class_name, confidence, bbox, light_color}
        annotated  : frame with YOLO bounding boxes drawn
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=frame,
        conf=conf_threshold,
        imgsz=320,
        device=device,
        half=(device == "cuda"),
        verbose=False,
    )

    detections = []
    annotated  = frame.copy()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id not in TARGET_CLASSES:
                continue

            confidence  = float(box.conf[0])
            class_name  = TARGET_CLASSES[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── Traffic light: crop and analyze color ──────────────────────
            light_color = "UNKNOWN"
            if class_name == "traffic light":
                fy1 = max(0, y1);  fy2 = min(frame.shape[0], y2)
                fx1 = max(0, x1);  fx2 = min(frame.shape[1], x2)
                crop = frame[fy1:fy2, fx1:fx2]
                light_color = get_traffic_light_color(crop)

            detections.append({
                "class_id"   : class_id,
                "class_name" : class_name,
                "confidence" : confidence,
                "bbox"       : (x1, y1, x2, y2),
                "light_color": light_color,
            })

            # ── Draw YOLO box ──────────────────────────────────────────────
            base_color = CLASS_COLORS.get(class_id, (255, 255, 255))
            bgr        = (base_color[2], base_color[1], base_color[0])

            # Override traffic-light color with detected state
            if class_name == "traffic light":
                if light_color == "RED":
                    bgr = (0, 0, 255)
                elif light_color == "GREEN":
                    bgr = (0, 255, 0)
                else:
                    bgr = (0, 200, 200)   # yellow-ish for unknown

            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

            txt = f"{class_name}"
            if class_name == "traffic light" and light_color != "UNKNOWN":
                txt += f" [{light_color}]"

            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(annotated, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return detections, annotated
