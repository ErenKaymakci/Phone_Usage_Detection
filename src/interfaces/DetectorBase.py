from abc import ABC, abstractmethod
import numpy as np

class DetectorBase(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Return format:
        [{"bbox": (x1,y1,x2,y2), "label": "phone", "confidence": 0.9}]
        """
        pass
