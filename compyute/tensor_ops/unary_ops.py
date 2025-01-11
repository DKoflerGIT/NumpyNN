"""Tensor unary operations."""

from typing import Optional

from ..tensors import ShapeLike, Tensor, unpack

__all__ = [
    "abs",
    "clip",
    "cos",
    "cosh",
    "exp",
    "fft1d",
    "fft2d",
    "histogram",
    "ifft1d",
    "ifft2d",
    "is_nan",
    "log",
    "log2",
    "log10",
    "real",
    "round",
    "sech",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]


def abs(x: Tensor) -> Tensor:
    """Computes the element-wise absolute value of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise absolute value.
    """
    return x.abs()


def clip(
    x: Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None
) -> Tensor:
    """Returns a tensor with values clipped to min_value and max_value.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    min_val : float, optional
        Lower bound. Defaults to ``None``. If ``None``, no clipping is performed on this edge.
    max_val : float
        Upper bound. Defaults to ``None``. If ``None``, no clipping is performed on this edge.

    Returns
    -------
    Tensor
        Tensor containing clipped values.
    """
    return Tensor(x.device.module.clip(x.data, min_val, max_val))


def cos(x: Tensor) -> Tensor:
    """Computes the element-wise cosine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise cosine.
    """
    return Tensor(x.device.module.cos(x.data))


def cosh(x: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic cosine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise hyperbolic cosine.
    """
    return Tensor(x.device.module.cosh(x.data))


def exp(x: Tensor) -> Tensor:
    """Computes the element-wise exponential of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise exponential.
    """
    return Tensor(x.device.module.exp(x.data))


def fft1d(x: Tensor, n: Optional[int] = None, dim: int = -1) -> Tensor:
    """Computes the 1D Fast Fourier Transform over a given dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : int, optional
        Length of the transformed dimension of the output. Defaults to ``None``.
    dim : int, optional
        Dimension over which to perform the operation. Defaults to ``-1``.

    Returns
    -------
    Tensor
        Complex tensor containing the 1D FFT.
    """
    return Tensor(x.device.module.fft.fft(x.data, n, dim))


def fft2d(
    x: Tensor, n: Optional[ShapeLike] = None, dims: tuple[int, int] = (-2, -1)
) -> Tensor:
    """Computes the 2D Fast Fourier Transform over given dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : ShapeLike, optional
        Shape (length of each transformed dimension) of the output. Defaults to ``None``.
    dim : tuple[int, int], optional
        Dimensions over which to perform the operation. Defaults to ``(-2, -1)``.

    Returns
    -------
    Tensor
        Complex tensor containing the 2D FFT.
    """
    return Tensor(x.device.module.fft.fft2(x.data, n, dims))


def ifft1d(x: Tensor, n: Optional[int] = None, dim: int = -1) -> Tensor:
    """Computes the inverse 1D Fast Fourier Transform over a given dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : int, optional
        Length of the transformed dimension of the output. Defaults to ``None``.
    dim : int, optional
        Dimension over which to perform the operation. Defaults to ``-1``.

    Returns
    -------
    Tensor
        Float tensor containing the inverse 1D FFT.
    """
    return Tensor(x.device.module.fft.ifft(x.data, n, dim))


def ifft2d(
    x: Tensor, s: Optional[ShapeLike] = None, dims: tuple[int, int] = (-2, -1)
) -> Tensor:
    """Computes the inverse 2D Fast Fourier Transform over given dims.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    n : ShapeLike, optional
        Shape (length of each transformed dimension) of the output. Defaults to ``None``.
    dim : tuple[int, int], optional
        Dimensions over which to perform the operation. Defaults to ``(-2, -1)``.

    Returns
    -------
    Tensor
        Float tensor containing the inverse 2D FFT.
    """
    return Tensor(x.device.module.fft.ifft2(x.data, s, dims))


def imag(x: Tensor) -> Tensor:
    """Returns the imaginary part of a complex tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing imaginary values.
    """
    return x.imag()


def is_nan(x: Tensor) -> Tensor:
    """Returns ``True`` if the element in a tensor is not a number.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Boolean tensor.
    """
    return Tensor(x.device.module.isnan(x.data))


def histogram(
    x: Tensor,
    bins: int | Tensor = 10,
    binrange: Optional[tuple[float, float]] = None,
    density: bool = False,
    weights: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Computes the histogram of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    bins : int | Tensor
        The number of bins to use for the histogram. Defaults to 10.
            - ``int``:defines the number of equal-width bins.
            - ``Tensor``: defines the sequence of bin edges including the rightmost edge.

    range : tuple[float, float], optional
        Defines the range of the bins. Defaults to ``None``.
    density : bool, optional
        Whether to compute the density instead of the count. Defaults to ``False``.
    weights : Tensor, optional
        Each value in input contributes its associated weight
        towards its bin’s result. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor containing the histogram values.
    Tensor
        Tensor containing the bin edges.
    """
    b = unpack(bins)
    w = weights.data if weights is not None else None
    hist, bin_edges = x.device.module.histogram(
        x.data, b, binrange, density=density, weights=w
    )
    return Tensor(hist), Tensor(bin_edges)


def log(x: Tensor) -> Tensor:
    """Computes the element-wise natural log of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise natural log.
    """
    return Tensor(x.device.module.log(x.data))


def log2(x: Tensor) -> Tensor:
    """Computes the element-wise log base 2 of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise log base 2.
    """
    return Tensor(x.device.module.log2(x.data))


def log10(x: Tensor) -> Tensor:
    """Computes the element-wise log base 10 of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise log base 10.
    """
    return Tensor(x.device.module.log10(x.data))


def real(x: Tensor) -> Tensor:
    """Returns the real part of a complex tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing real values.
    """
    return x.real()


def round(x: Tensor, decimals: int) -> Tensor:
    """Rounds tensor elements.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    decimals : int
        Decimal places of rounded values.

    Returns
    -------
    Tensor
        Tensor containing rounded elements.
    """
    return Tensor(x.data.round(decimals))


def sech(x: Tensor) -> Tensor:
    """Computes the element-wise secant of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise secant.
    """
    return 1 / cosh(x)


def sin(x: Tensor) -> Tensor:
    """Computes the element-wise sine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise sine.
    """
    return Tensor(x.device.module.sin(x.data))


def sinh(x: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic sine of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise hyperbolic sine.
    """
    return Tensor(x.device.module.sinh(x.data))


def sqrt(x: Tensor) -> Tensor:
    """Computes the element-wise square root of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise square root.
    """
    return Tensor(x.device.module.sqrt(x.data))


def tan(x: Tensor) -> Tensor:
    """Computes the element-wise tangent of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise tangent.
    """
    return Tensor(x.device.module.tan(x.data))


def tanh(x: Tensor) -> Tensor:
    """Computes the element-wise hyperbolic tangent of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing the element-wise hyperbolic tangent.
    """
    return Tensor(x.device.module.tanh(x.data))
