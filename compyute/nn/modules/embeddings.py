"""Neural network embedding modules."""

from typing import Optional

from ...random import normal
from ...tensor_ops.creation_ops import empty
from ...tensors import Tensor
from ..functional.embedding_funcs import EmbeddingFunction
from ..parameter import Parameter
from .module import Module

__all__ = ["Embedding"]


class Embedding(Module):
    r"""Lookup embedding layer.

    Shapes:
        - Input :math:`(B_1, ... , B_n, S)`
        - Output :math:`(B_1, ... , B_n, S, E)`
    where
        - :math:`B_1, ... , B_n` ... batch dimensions
        - :math:`S` ... sequence
        - :math:`E` ... embedding dimension

    Parameters
    ----------
    n_embeds : int
        Number of embedding vectors.
    embed_dim : int
        Nubmer of embedding vector dimensions.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Embeddings are initialized from :math:`\mathcal{N}(0, 1)`.
    """

    def __init__(
        self, n_embeds: int, embed_dim: int, label: Optional[str] = None
    ) -> None:
        super().__init__(label)
        self.n_embeddings = n_embeds
        self.embedding_dim = embed_dim

        # init parameters
        self.w = Parameter(normal((n_embeds, embed_dim)))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return EmbeddingFunction.forward(self.function_ctx, x, self.w)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dw = EmbeddingFunction.backward(self.function_ctx, dy)
        self.update_parameter_grad(self.w, dw)
        return empty((0,))
