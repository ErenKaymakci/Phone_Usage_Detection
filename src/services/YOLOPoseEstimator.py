import numpy as np
from ultralytics import YOLO
from interfaces.DetectorBase import DetectorBase

class YOLOPoseEstimator(DetectorBase):
    def __init__(self, model_path: str, conf: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray):
        results = self.model(frame, conf=self.conf)
        detections = []
        if results and results[0].keypoints is not None:
            for idx, box in enumerate(results[0].boxes):
                cls = results[0].names[int(box.cls)]
                if cls == "person":  # sadece person
                    detections.append({
                        "bbox": box.xyxy[0].tolist(),
                        "keypoints": results[0].keypoints[idx].xy.cpu().numpy()[0].tolist(),
                        "confidence": float(box.conf)
                    })
        return detections
