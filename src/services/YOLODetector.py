import numpy as np
from ultralytics import YOLO
from interfaces.DetectorBase import DetectorBase

class YOLODetector(DetectorBase):
    def __init__(self, model_path: str, conf: float = 0.5, classes: list[str] = None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.allowed_classes = set(classes) if classes else None
        
    def detect(self, frame: np.ndarray, rois: list[dict] = None) -> list[dict]:
        detections = []
        if rois is None:
            # Normal: tüm frame
            results = self.model(frame, conf=self.conf, imgsz=1280)
            detections += self._parse_results(results, frame, None)
        else:
            # Sadece ROI (ör: kişiler)
            for roi in rois:
                x1, y1, x2, y2 = map(int, roi["bbox"])
                crop = frame[y1:y2, x1:x2]
                results = self.model(crop, conf=self.conf, imgsz=640)

                detections += self._parse_results(results, frame, roi["bbox"])
        return detections


    def _parse_results(self, results, full_frame, parent_bbox=None):
        dets = []
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls)]
            if self.allowed_classes is None or cls in self.allowed_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Eğer crop'tan geldiyse, koordinatları global frame'e geri map et
                if parent_bbox:
                    px1, py1, _, _ = map(int, parent_bbox)
                    x1, x2 = x1 + px1, x2 + px1
                    y1, y2 = y1 + py1, y2 + py1

                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": cls,
                    "confidence": float(box.conf)
                })
        return dets

