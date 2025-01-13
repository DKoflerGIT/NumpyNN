"""Neural network embedding functions."""

from ...tensor_ops.creation_ops import zeros
from ...tensor_ops.multiary_ops import add_at
from ...tensors import Tensor
from ...typing import is_integer
from .functions import Function, FunctionContext, PseudoContext

__all__ = ["embedding"]


class EmbeddingFunction(Function):
    """Performs lookup embedding on a tensor of indices."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, embed_table: Tensor) -> Tensor:
        if not is_integer(x.dtype):
            raise ValueError(f"Input must be an integer, got '{x.dtype}'.")
        y = embed_table[x]
        ctx.add(x, embed_table.shape)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x, emb_shape = ctx.get()
        dw = zeros(emb_shape, device=dy.device, dtype=dy.dtype)
        add_at(dw, x, dy)
        return dw


def embedding(x: Tensor, embed_table: Tensor) -> Tensor:
    """Performs lookup embedding using a tensor of integer indices.

    Parameters
    ----------
    x : Tensor
        Input tensor containing indices. Must be integers.
    emb_table : Tensor
        Tensor of embeddings.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return EmbeddingFunction.forward(PseudoContext(), x, embed_table)
