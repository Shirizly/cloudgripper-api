from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class IRobot(ABC):
    @abstractmethod
    async def move_xy(self, position: Tuple[float, float]) -> None:
        pass

    @abstractmethod
    async def move_z(self, height: float) -> None:
        pass

    @abstractmethod
    async def gripper_open(self) -> None:
        pass

    @abstractmethod
    async def gripper_close(self) -> None:
        pass

    @abstractmethod
    async def get_bottom_image(self) -> np.ndarray:
        pass
