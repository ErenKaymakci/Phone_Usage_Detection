from abc import ABC, abstractmethod
import numpy as np

class FrameSourceBase(ABC):
    @abstractmethod
    def open(self, path: str) -> None:
        pass

    @abstractmethod
    def read_frame(self) -> tuple[bool, np.ndarray]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass
