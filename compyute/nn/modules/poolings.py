"""Neural network pooling modules."""

from typing import Optional

from ...tensors import ShapeLike, Tensor
from ..functional.pooling_funcs import (
    AvgPooling2DFunction,
    MaxPooling2DFunction,
    Upsample2DFunction,
)
from .module import Module

__all__ = ["Upsample2D", "MaxPooling2D", "AvgPooling2D"]


class Upsample2D(Module):
    """Layer used for upsamling by repeating values over the last two dimensions.

    Parameters
    ----------
    scaling : int, optional
        Scaling factor for the upsampling. Defaults to ``2``.
    target_shape : ShapeLike, optional
        Shape of the target tensor. Defaults to ``None``. If not ``None`` and
        shapes do not match after upsampling, remaining values are filled with zeroes.
    """

    def __init__(
        self,
        scaling: int = 2,
        target_shape: Optional[ShapeLike] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.scaling = scaling
        self.target_shape = target_shape

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return Upsample2DFunction.forward(self.ctx, x, self.scaling, self.target_shape)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return Upsample2DFunction.backward(self.ctx, dy)


class MaxPooling2D(Module):
    """Pooling layer used for downsampling where the
    maximum value within the pooling window is used.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the pooling window used for the pooling operation. Defaults to ``2``.
    """

    def __init__(self, kernel_size: int = 2, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.kernel_size = kernel_size

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return MaxPooling2DFunction.forward(self.ctx, x, self.kernel_size)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return MaxPooling2DFunction.backward(self.ctx, dy)


class AvgPooling2D(Module):
    """Pooling layer used for downsampling where the
    average value within the pooling window is used.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the pooling window used for the pooling operation. Defaults to ``2``.
    """

    def __init__(self, kernel_size: int = 2, label: Optional[str] = None) -> None:
        super().__init__(label)
        self.kernel_size = kernel_size

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return AvgPooling2DFunction.forward(self.ctx, x, self.kernel_size)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return AvgPooling2DFunction.backward(self.ctx, dy)
