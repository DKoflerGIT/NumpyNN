"""Neural network embedding modules."""

from typing import Optional

from ...base_tensor import Tensor
from ...dtypes import Dtype, _DtypeLike
from ...random.random import normal
from ...tensor_ops.creating import zeros_like
from ..functional.embeddings import lookup_embedding
from ..parameter import Parameter
from .module import Module

__all__ = ["Embedding"]


class Embedding(Module):
    r"""Lookup embedding layer.

    Shapes:
        - Input :math:`(B_1, ... , B_n, S)`
        - Output :math:`(B_1, ... , B_n, S, E)`
    where
        - :math:`B_1, ... , B_n` ... batch axes
        - :math:`S` ... sequence
        - :math:`E` ... embedding dimension

    Parameters
    ----------
    n_embeddings : int
        Number of embedding vectors.
    embedding_dim : int
        Embedding vector dimensions.
    dtype : _DtypeLike, optional
        Datatype of weights and biases. Defaults to :class:`compyute.float32`.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Embeddings are initialized from :math:`\mathcal{N}(0, 1)`.
    """

    def __init__(
        self, n_embeddings: int, embedding_dim: int, dtype: _DtypeLike = Dtype.FLOAT32, label: Optional[str] = None
    ) -> None:
        super().__init__(label)
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim

        # init weights
        self.w = Parameter(normal((n_embeddings, embedding_dim), dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        y, grad_fn = lookup_embedding(x, self.w, self._is_training)

        if self._is_training:

            def _backward(dy: Tensor) -> Tensor:
                self._update_parameter_grad(self.w, grad_fn(dy))
                return zeros_like(x)

            self._backward = _backward

        return y
