"""Neural network shape changing modules."""

from typing import Optional

from ...tensors import ShapeLike, Tensor
from ..functional.shape_funcs import FlattenFn, ReshapeFn, SliceFn
from .module import Module

__all__ = ["Flatten", "Reshape", "Slice"]


class Flatten(Module):
    """Flattes tensors not including the batch dimension.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FlattenFn.forward(self.fcache, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FlattenFn.backward(self.fcache, dy)


class Reshape(Module):
    """Reshapes tensors.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the output tensor.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, shape: ShapeLike, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.shape = shape

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return ReshapeFn.forward(self.fcache, x, self.shape)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return ReshapeFn.backward(self.fcache, dy)


class Slice(Module):
    """Slices tensors.

    Parameters
    ----------
    slice : str
        Slice of the output tensor as a string.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, slice: str, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.slice = _parse_slices(slice)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return SliceFn.forward(self.fcache, x, self.slice)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return SliceFn.backward(self.fcache, dy)


def _parse_slices(slice_str: str) -> tuple[slice | int, ...]:
    slice_parts = slice_str.split(",")

    slices: list[slice | int] = []
    for part in slice_parts:
        if ":" in part:
            sub_parts = part.strip().split(":")
            start = int(sub_parts[0]) if sub_parts[0] else None
            stop = int(sub_parts[1]) if len(sub_parts) > 1 and sub_parts[1] else None
            step = int(sub_parts[2]) if len(sub_parts) > 2 and sub_parts[2] else None
            slices.append(slice(start, stop, step))
        else:
            slices.append(int(part))

    return tuple(slices)
