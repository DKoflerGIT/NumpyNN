"""Neural network shape changing functions."""

from ...tensor_ops.creation_ops import zeros
from ...tensors import ShapeLike, Tensor
from .functions import Function, FunctionContext


class FlattenFunction(Function):
    """Flattens tensors not including the batch dimension."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor) -> Tensor:
        ctx.add(x.shape)
        return x.view((x.shape[0], -1))

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x_shape = ctx.get()
        return dy.view(x_shape)


class ReshapeFunction(Function):
    """Reshapes tensors."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, shape: ShapeLike) -> Tensor:
        ctx.add(x.shape)
        return x.view((x.shape[0],) + shape)

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x_shape = ctx.get()
        return dy.view(x_shape)


class SliceFunction(Function):
    """Slices tensors."""

    @staticmethod
    def forward(
        ctx: FunctionContext, x: Tensor, slice: tuple[slice | int, ...]
    ) -> Tensor:
        ctx.add(x.shape, slice)
        return x[slice]

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x_shape, slice = ctx.get()
        dx = zeros(x_shape, device=dy.device, dtype=dy.dtype)
        dx[slice] = dy
        return dx
