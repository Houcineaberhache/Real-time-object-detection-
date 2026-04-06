import cv2
from ultralytics import YOLO
import pyttsx3
import time
from interaction_module import InteractionManager  # <--- MISSING LINE 1

# 1. Setup Voice
# We initialize the manager you built instead of just the engine
ui_manager = InteractionManager()                # <--- MISSING LINE 2
last_alert_time = 0

# 2. Load the Model
model = YOLO('yolov8n.pt') 

# 3. Process Video
cap = cv2.VideoCapture('Video.mp4')

while cap.isOpened():
    start_frame = time.time()
    success, frame = cap.read()
    
    if success:
        # Run detection
        results = model(frame, conf=0.5)
        
        # Plot results
        annotated_frame = results[0].plot()

        # YOUR PART: Voice Alert Logic
        for box in results[0].boxes:
            label = model.names[int(box.cls)]
            
            # Using your ui_manager to handle the voice and cooldown
            if label in ['person', 'stop sign', 'traffic light']:
                ui_manager.trigger_voice_alert(label)

        # YOUR PART: UI Overlay (FPS)
        end_frame = time.time()
        fps = 1 / (end_frame - start_frame)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Smart Driving Assistant", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()