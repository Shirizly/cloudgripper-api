from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autograsper import Autograsper  # For type hinting without circular imports


class Task(ABC):
    @abstractmethod
    async def execute(self, autograsper: 'Autograsper') -> None:
        """Executes the task using the provided autograsper."""
        pass

    @abstractmethod
    async def detect_errors(self, autograsper: 'Autograsper') -> bool:
        """Detects if any errors occurred during task execution."""
        pass

    @abstractmethod
    async def recover_from_errors(self, autograsper: 'Autograsper') -> None:
        """Recovers from any errors detected."""
        pass

    @abstractmethod
    async def reset_scene(self, autograsper: 'Autograsper') -> None:
        """Resets the scene after task completion or errors."""
        pass
