"""Neural network linear functions."""

from typing import Optional

from ...tensors import Tensor
from .functions import Function, FunctionContext, PseudoContext

__all__ = ["linear"]


class LinearFunction(Function):
    """Applies a linear transformation to the input."""

    @staticmethod
    def forward(
        ctx: FunctionContext, x: Tensor, w: Tensor, b: Optional[Tensor]
    ) -> Tensor:
        y = x @ w.T
        if b:
            y += b

        ctx.add(x, w, b is not None)
        return y

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        x, w, b = ctx.get()

        dx = dy @ w
        dw = (dy.T @ x).sum(tuple(range(dy.ndim - 2)))
        db = None if not b else dy.sum(tuple(range(dy.ndim - 1)))

        return dx, dw, db


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Applies a linear transformation to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None``. If ``None``, no bias is added.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Linear`
    """
    return LinearFunction.forward(PseudoContext(), x, w, b)
