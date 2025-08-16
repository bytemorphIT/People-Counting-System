
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD

class Detector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        return results
