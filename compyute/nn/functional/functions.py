"""Neural network function"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

__all__ = ["Function", "FunctionContext", "no_cache_ctx"]


class FunctionContext:
    """Cache for context data that needs to be cached for gradient computation.
    Items are cached as a stack (last in - first out)"""

    context: deque[tuple[Any, ...]]

    def __init__(self) -> None:
        self.context = deque()

    def add(self, *items: Any) -> None:
        """Adds items to the function context stack."""
        if get_cache_ctx_enabled():
            self.context.append(items)

    def get(self) -> Any:
        """Removes and returns the topmost items from the function context stack."""
        items = self.context.pop()
        if len(items) == 1:
            return items[0]
        return items


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


context_enabled: bool = True


def get_cache_ctx_enabled() -> bool:
    """Returns ``True`` if caching of context data for gradient computation is enabled."""
    return context_enabled


def set_cache_ctx_enabled(enabled: bool) -> None:
    """Sets whether caching of context data for gradient computation is enabled."""
    global context_enabled
    context_enabled = enabled


@contextmanager
def no_cache_ctx() -> Generator:
    """Context manager to disable caching of context data for gradient computation."""
    set_cache_ctx_enabled(False)
    try:
        yield
    finally:
        set_cache_ctx_enabled(True)
