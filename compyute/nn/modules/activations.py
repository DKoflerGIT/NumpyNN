"""Activation modules."""

from typing import Literal, Optional, TypeAlias

from ...tensors import Tensor
from ..functional.activation_funcs import (
    FastGELUFunction,
    GELUFunction,
    LeakyReLUFunction,
    ReLUFunction,
    SigmoidFunction,
    SiLUFunction,
    SoftmaxFunction,
    TanhFunction,
)
from .module import Module

__all__ = [
    "ReLU",
    "LeakyReLU",
    "GELU",
    "FastGELU",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Tanh",
]


class GELU(Module):
    r"""Gaussian Error Linear Unit activation function (using the :math:`tanh` approximation)
    as described by `Hendrycks et al., 2016 <https://arxiv.org/pdf/1606.08415>`_.

    .. math::
        y = 0.5 \cdot x \cdot \left( 1 + \text{tanh} \left( x \sqrt{\frac{2}{\pi}} \cdot (1 + 0.044715 \cdot x^2) \right) \right)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return GELUFunction.forward(self.ctx, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return GELUFunction.backward(self.ctx, dy)


class FastGELU(Module):
    r"""Gaussian Error Linear Unit activation function (using the :math:`sigmoid` approximation)
    as described by `Hendrycks et al., 2016 <https://arxiv.org/pdf/1606.08415>`_.

    .. math::
        y = x \cdot \sigma{1.702x}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return FastGELUFunction.forward(self.ctx, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return FastGELUFunction.backward(self.ctx, dy)


class LeakyReLU(Module):
    r"""Leaky ReLu activation function.

    .. math::
        y = \text{max}(\alpha \cdot x, x)

    Parameters
    ----------
    alpha : float, optional
        Slope of the negative output. Defaults to ``0.01``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, alpha: float = 0.01, label: Optional[str] = None):
        super().__init__(label)
        self.alpha = alpha

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return LeakyReLUFunction.forward(self.ctx, x, self.alpha)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return LeakyReLUFunction.backward(self.ctx, dy)


class ReLU(Module):
    r"""Applies the Rectified Linear Unit activation function to an input tensor.

    .. math::
        y = \text{max}(0, x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.

    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return ReLUFunction.forward(self.ctx, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return ReLUFunction.backward(self.ctx, dy)


class Sigmoid(Module):
    r"""Sigmoid activation function.

    .. math::
        y = \frac{1}{1 + e^{-x}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return SigmoidFunction.forward(self.ctx, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return SigmoidFunction.backward(self.ctx, dy)


class SiLU(Module):
    r"""Sigmoid Linear Unit activation function.

    .. math::
        y = x \cdot \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return SiLUFunction.forward(self.ctx, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return SiLUFunction.backward(self.ctx, dy)


class Softmax(Module):
    r"""Softmax activation function.

    .. math::
        y = \frac{e^x}{\sum_{i=1}^N e^{x_i}}

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return SoftmaxFunction.forward(self.ctx, x, -1)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return SoftmaxFunction.backward(self.ctx, dy)


class Tanh(Module):
    r"""Tanh activation function.

    .. math::
        y = \text{tanh}(x)

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return TanhFunction.forward(self.ctx, x)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return TanhFunction.backward(self.ctx, dy)


ActivationLike: TypeAlias = Literal[
    "relu", "leaky_relu", "gelu", "sigmoid", "silu", "tanh"
]
ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "gelu": GELU,
    "sigmoid": Sigmoid,
    "silu": SiLU,
    "tanh": Tanh,
}


def get_activation(activation: ActivationLike) -> Module:
    """Returns an actiation function."""

    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation function: {activation}.")
    return ACTIVATIONS[activation]()
