from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Validate the model
# This will automatically download a small part of the COCO dataset to test
metrics = model.val(data='coco8.yaml')

print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")