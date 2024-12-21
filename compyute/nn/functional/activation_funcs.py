"""Neural network activation functions."""

from ...tensor_ops.selection_ops import maximum
from ...tensor_ops.unary_ops import exp
from ...tensor_ops.unary_ops import tanh as _tanh
from ...tensors import Tensor
from .functions import Function, FunctionContext, PseudoContext

__all__ = [
    "relu",
    "leaky_relu",
    "gelu",
    "fast_gelu",
    "sigmoid",
    "silu",
    "tanh",
    "softmax",
]


class ReLUFunction(Function):
    """Applies the Rectified Linear Unit activation function to the input."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor) -> Tensor:
        y = maximum(x, 0.0)
        ctx.add(y > 0.0)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        mask = ctx.get()
        return dy * mask


def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit activation function to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    ----------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.ReLU`
    """
    return ReLUFunction.forward(PseudoContext(), x)


class LeakyReLUFunction(Function):
    """Applies the leaky ReLU function to an input tensor."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, alpha: float) -> Tensor:
        y = maximum(alpha * x, x)
        ctx.add(alpha, y > 0.0)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        alpha, mask = ctx.get()
        return dy * (mask + (~mask).to_type(dy.dtype) * alpha)


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Applies the leaky ReLU function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float, optional
        Slope of the negative output. Defaults to ``0.01``.

    Returns
    ----------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.LeakyReLU`
    """
    return LeakyReLUFunction.forward(PseudoContext(), x, alpha)


class SigmoidFunction(Function):
    """Applies the sigmoid function to an input tensor."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor) -> Tensor:
        y = 1.0 / (1.0 + exp(-x))
        ctx.add(y)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        y = ctx.get()
        return y * (1.0 - y) * dy


def sigmoid(x: Tensor) -> Tensor:
    """Applies the sigmoid function to an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.Sigmoid`
    """
    return SigmoidFunction.forward(PseudoContext(), x)


class TanhFunction(Function):
    """Applies the hyperbolic tangent activation function to the input."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor) -> Tensor:
        y = _tanh(x)
        ctx.add(y)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        y = ctx.get()
        return (1.0 - y * y) * dy


def tanh(x: Tensor) -> Tensor:
    """Applies the hyperbolic tangent activation function to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.Tanh`
    """
    return TanhFunction.forward(PseudoContext(), x)


class GELUFunction(Function):
    """Applies the Gaussian Error Linear Unit activation function to the input."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor) -> Tensor:
        # sqrt(2/pi) = 0.7978845608
        tanh_term = _tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        y = 0.5 * x * (1.0 + tanh_term)
        ctx.add(x, tanh_term)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x, tanh_term = ctx.get()
        dx1 = 1.0 + tanh_term
        # sqrt(2/pi) * 3 * 0.044715 = 0.1070322243
        dx2 = x * (1.0 - tanh_term * tanh_term) * (0.7978845608 + 0.1070322243 * x * x)
        return 0.5 * dy * (dx1 + dx2)


def gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Unit activation function to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.GELU`
    """
    return GELUFunction.forward(PseudoContext(), x)


class FastGELUFunction(Function):
    """Applies the Gaussian Error Linear Unit activation function to the input."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor) -> Tensor:
        sigm = 1.0 / (1.0 + exp(x * -1.702))
        y = x * sigm
        ctx.add(x, sigm)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x, sigm = ctx.get()
        return dy * sigm * (1.0 + x * 1.702 * (1.0 - sigm))


def fast_gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Unit activation function to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.FastGELU`
    """
    return FastGELUFunction.forward(PseudoContext(), x)


class SiLUFunction(Function):
    """Applies the Sigmoid Linear Unit activation function to the input."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor) -> Tensor:
        sigm = 1.0 / (1.0 + exp(-x))
        y = x * sigm
        ctx.add(x, sigm)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        x, sigm = ctx.get()
        return dy * sigm * (1.0 + x * (1.0 - sigm))


def silu(x: Tensor) -> Tensor:
    """Applies the Sigmoid Linear Unit activation function to the input.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.SiLU`
    """
    return SiLUFunction.forward(PseudoContext(), x)


class SoftmaxFunction(Function):
    """Applies the softmax activation function to the last dimension of an input tensor."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, dim: int) -> Tensor:
        x = exp(x - x.max(dim, keepdims=True))
        y = x / x.sum(dim, keepdims=True)
        ctx.add(dim, y)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        dim, y = ctx.get()
        return y * (dy - (dy * y).sum(dim, keepdims=True))  # thank you ChatGPT


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Applies the softmax activation function to the last dimension of an input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int, optional
        Dimension to apply softmax to. Defaults to ``-1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    --------
    :class:`compyute.nn.Softmax`
    """
    return SoftmaxFunction.forward(PseudoContext(), x, dim)
