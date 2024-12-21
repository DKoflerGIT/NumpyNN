"""Neural network pooling functions."""

from typing import Optional

from ...tensor_ops.shape_ops import pad_to_shape, pooling2d, repeat2d
from ...tensors import ShapeError, ShapeLike, Tensor
from .functions import Function, FunctionContext, PseudoContext

__all__ = ["upsample2d", "maxpooling2d", "avgpooling2d"]


class Upsample2DFunction(Function):
    """Upsamples a tensor by repeating values over the last two dimensions."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        scaling: int,
        target_shape: Optional[ShapeLike],
    ) -> Tensor:
        # if x.ndim != 4:
        #     raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")

        y = repeat2d(x, scaling)
        if target_shape is not None and y.shape != target_shape:
            y = pad_to_shape(y, target_shape)

        ctx.add(scaling)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        scaling = ctx.get()
        return pooling2d(dy, scaling, scaling).sum((-2, -1))


def upsample2d(
    x: Tensor,
    scaling: int = 2,
    target_shape: Optional[ShapeLike] = None,
) -> Tensor:
    """Upsamples a tensor by repeating values over the last two dimensions.

    Parameters
    ----------
    x : Tensor
        Tensor to be upsampled.
    scaling : int, optional
        Scaling factor for the upsampling. Defaults to ``2``.
    target_shape : ShapeLike, optional
        Shape of the target tensor. Defaults to ``None``. If not ``None`` and
        shapes do not match after upsampling, remaining values are filled with zeroes.

    Returns
    -------
    Tensor
        Upsampled tensor.
    """
    return Upsample2DFunction.forward(PseudoContext(), x, scaling, target_shape)


class MaxPooling2DFunction(Function):
    """Performs max pooling over the last two dimensions."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, kernel_size: int) -> Tensor:
        if x.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")
        y = pooling2d(x, kernel_size, kernel_size).max((-2, -1))
        ctx.add(x, kernel_size, y)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x, kernel_size, y = ctx.get()
        mask = upsample2d(y, kernel_size, x.shape) == x
        return upsample2d(dy, kernel_size, x.shape) * mask


def maxpooling2d(x: Tensor, kernel_size: int = 2) -> Tensor:
    """Performs max pooling over the last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : int, optional
        Size of the pooling window. Defaults to ``2``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.MaxPooling2D`
    """
    return MaxPooling2DFunction.forward(PseudoContext(), x, kernel_size)


class AvgPooling2DFunction(Function):
    """Performs average pooling over the last two dimensions."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, kernel_size: int) -> Tensor:
        if x.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")
        y = pooling2d(x, kernel_size, kernel_size).mean((-2, -1))
        ctx.add(x.shape, kernel_size)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x_shape, kernel_size = ctx.get()
        return upsample2d(dy / (kernel_size * kernel_size), kernel_size, x_shape)


def avgpooling2d(x: Tensor, kernel_size: int = 2) -> Tensor:
    """Performs average pooling over the last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : int, optional
        Size of the pooling window. Defaults to ``2``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.AvgPooling2D`
    """
    return AvgPooling2DFunction.forward(PseudoContext(), x, kernel_size)
