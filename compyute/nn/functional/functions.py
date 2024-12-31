"""Neural network function"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

__all__ = ["Function", "FunctionContext"]


class FunctionContext:
    """Cache for context data that needs to be cached for gradient computation.
    Items are cached as a stack (last in - first out)"""

    context: deque[tuple[Any, ...]]

    def __init__(self) -> None:
        self.context = deque()

    def add(self, *items: Any) -> None:
        """Adds items to the function context stack."""
        self.context.append(items)

    def get(self) -> Any:
        """Removes and returns the topmost items from the function context stack."""
        items = self.context.pop()
        if len(items) == 1:
            return items[0]
        return items

    def clear(self) -> None:
        """Empties the context cache."""
        self.context.clear()


class PseudoContext(FunctionContext):
    """Pseudo context that does not cache anything."""

    def add(self, *values: Any) -> None: ...


class Function(ABC):
    """Neural network function base class."""

    def __init__(self) -> None:
        raise NotImplementedError("Function cannot be instantiated.")

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Use ``forward()`` instead.")

    @staticmethod
    @abstractmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        """Forward pass of the function."""

    @staticmethod
    @abstractmethod
    def backward(*args: Any, **kwargs: Any) -> Any:
        """Backward pass of the function."""
