import cv2
import numpy as np
from interfaces.FrameSourceBase import FrameSourceBase

class FrameSourceCV(FrameSourceBase):
    def __init__(self):
        self.cap = None

    def open(self, path: str) -> None:
        self.cap = cv2.VideoCapture(path)

    def read_frame(self) -> tuple[bool, np.ndarray]:
        if self.cap:
            return self.cap.read()
        return False, None

    def release(self) -> None:
        if self.cap:
            self.cap.release()
