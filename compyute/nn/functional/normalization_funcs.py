"""Neural network normalization functions."""

from ...tensor_ops.unary_ops import sqrt
from ...tensors import ShapeError, Tensor
from .functions import Function, FunctionContext, PseudoContext

__all__ = ["batchnorm1d", "batchnorm2d", "layernorm", "rmsnorm"]


class BatchNorm1DFunction(Function):
    """Performs 1D batch normalization on a tensor."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        rmean: Tensor,
        rvar: Tensor,
        w: Tensor,
        b: Tensor,
        m: float,
        eps: float,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim not in {2, 3}:
            raise ShapeError(f"Expected input to be 2D or 3D, got {x.ndim}D.")

        x_is_2d = x.ndim == 2
        batch_dims: tuple[int, ...] = (0,) if x.ndim == 2 else (0, 2)

        if training:
            # compute mean and variance from x
            mean = x.mean(batch_dims, keepdims=True)
            std = sqrt(x.var(batch_dims, keepdims=True) + eps)
            x_norm = (x - mean) / std

            # keep running stats
            rmean = rmean * (1 - m) + mean.squeeze() * m
            rvar = rvar * (1 - m) + x.var(batch_dims, ddof=1) * m
        else:
            # use running mean and variance
            var = rvar if x_is_2d else rvar.view((*rvar.shape, 1))
            mean = rmean if x_is_2d else rmean.view((*rmean.shape, 1))
            std = sqrt(var + eps)
            x_norm = (x - mean) / std

        w = w if x_is_2d else w.view((*w.shape, 1))
        b = b if x_is_2d else b.view((*b.shape, 1))
        y = w * x_norm + b

        ctx.add(w, batch_dims, std, x_norm)
        return y, rmean, rvar

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        w, batch_dims, std, x_norm = ctx.get()
        n = float(dy.size / dy.shape[1])

        # input grads
        dy_sum = dy.sum(batch_dims, keepdims=True)
        dy_x_norm_sum = (dy * x_norm).sum(batch_dims, keepdims=True)
        dx = w / (std * n) * (n * dy - dy_sum - x_norm * dy_x_norm_sum)

        # gamma grads
        dw = dy_x_norm_sum.squeeze()

        # beta grads
        db = dy_sum.squeeze()

        return dx, dw, db


def batchnorm1d(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    training: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Performs 1D batch normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    rmean : Tensor
        Running mean tensor.
    rvar : Tensor
        Running variance tensor.
    w : Tensor
        Weight tensor for scaling the distribution.
    b : Tensor
        Bias tensor for shifting the distribution.
    m : float, optional
        Momentum used for running mean and variance computation. Defaults to ``0.1``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    training : bool, optional
        Whether to perform calculations in training mode. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Tensor
        New running mean.
    Tensor
        New running variance.

    See Also
    ----------
    :class:`compyute.nn.BatchNorm1D`
    """
    return BatchNorm1DFunction.forward(
        PseudoContext(), x, rmean, rvar, w, b, m, eps, training
    )


class BatchNorm2DFunction(Function):
    """Performs 2D batch normalization on a tensor."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        rmean: Tensor,
        rvar: Tensor,
        w: Tensor,
        b: Tensor,
        m: float,
        eps: float,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")
        batch_dims = (0, 2, 3)

        if training:
            # compute mean and variance from x
            mean = x.mean(batch_dims, keepdims=True)
            std = sqrt(x.var(batch_dims, keepdims=True) + eps)
            x_norm = (x - mean) / std

            # keep running stats
            rmean = rmean * (1 - m) + mean.squeeze() * m
            rvar = rvar * (1 - m) + x.var(batch_dims, ddof=1) * m
        else:
            # use running mean and variance
            mean = rmean.view((*rmean.shape, 1, 1))
            std = sqrt(rvar.view((*rvar.shape, 1, 1)) + eps)
            x_norm = (x - mean) / std

        w = w.view((*w.shape, 1, 1))
        b = b.view((*b.shape, 1, 1))
        y = w * x_norm + b

        ctx.add(w, batch_dims, std, x_norm)
        return y, rmean, rvar

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        w, batch_dims, std, x_norm = ctx.get()
        n = float(dy.size / dy.shape[1])

        # input grads
        dy_sum = dy.sum(batch_dims, keepdims=True)
        dy_x_norm_sum = (dy * x_norm).sum(batch_dims, keepdims=True)
        dx = w / (std * n) * (n * dy - dy_sum - x_norm * dy_x_norm_sum)

        # gamma grads
        dw = dy_x_norm_sum.squeeze()

        # beta grads
        db = dy_sum.squeeze()

        return dx, dw, db


def batchnorm2d(
    x: Tensor,
    rmean: Tensor,
    rvar: Tensor,
    w: Tensor,
    b: Tensor,
    m: float = 0.1,
    eps: float = 1e-5,
    training: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Performs 2D batch normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    rmean : Tensor
        Running mean values.
    rvar : Tensor
        Running variance values.
    w : Tensor
        Weight tensor for scaling the distribution.
    b : Tensor
        Bias tensor for shifting the distribution.
    m : float, optional
        Momentum used for running mean and variance computation. Defaults to ``0.1``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.
    training : bool, optional
        Whether to perform calculations in training mode. Defaults to ``False``.

    Returns
    -------
    Tensor
        Output tensor.
    Tensor
        New running mean.
    Tensor
        New running variance.

    See Also
    ----------
    :class:`compyute.nn.BatchNorm2D`
    """
    return BatchNorm2DFunction.forward(
        PseudoContext(), x, rmean, rvar, w, b, m, eps, training
    )


class LayerNormFunction(Function):
    """Performs layer normalization on a tensor."""

    @staticmethod
    def forward(
        ctx: FunctionContext, x: Tensor, w: Tensor, b: Tensor, eps: float
    ) -> Tensor:
        feat_dims = tuple(-i - 1 for i in range(w.ndim))

        mean = x.mean(feat_dims, keepdims=True)
        std = sqrt(x.var(feat_dims, keepdims=True) + eps)
        x_norm = (x - mean) / std
        y = w * x_norm + b

        ctx.add(w, feat_dims, std, x_norm)
        return y

    @staticmethod
    def backward(ctc: FunctionContext, dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        w, feat_dims, std, x_norm = ctc.get()
        batch_dims = tuple(range(dy.ndim - w.ndim))
        n = w.size

        # input grads
        dy_sum = dy.sum(feat_dims, keepdims=True)
        dy_x_norm = dy * x_norm
        dy_x_norm_sum = dy_x_norm.sum(feat_dims, keepdims=True)
        dx = w / (std * n) * (n * dy - dy_sum - x_norm * dy_x_norm_sum)

        # gamma grads
        dw = dy_x_norm.sum(batch_dims)

        # beta grads
        db = dy.sum(batch_dims)

        return dx, dw, db


def layernorm(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    """Performs layer normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor for scaling the distribution.
    b : Tensor
        Bias tensor for shifting the distribution.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.LayerNorm`
    """
    return LayerNormFunction.forward(PseudoContext(), x, w, b, eps)


class RMSNormFunction(Function):
    """Performs RMS normalization on a tensor."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, w: Tensor, eps: float) -> Tensor:
        feat_dims = tuple(-i - 1 for i in range(w.ndim))

        rms = sqrt((x * x).mean(feat_dims, keepdims=True) + eps)
        x_norm = x / rms
        y = w * x_norm

        ctx.add(x, w, feat_dims, rms, x_norm)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> tuple[Tensor, Tensor]:
        x, w, feat_dims, rms, x_norm = ctx.get()
        sum_dims = tuple(range(x.ndim - w.ndim))

        # input grads
        dy_x_sum = (dy * x).sum(feat_dims, keepdims=True)
        dx = w / rms * (dy - x_norm * dy_x_sum / (w.size * rms))

        # gamma grads
        dw = (dy * x_norm).sum(sum_dims)

        return dx, dw


def rmsnorm(x: Tensor, w: Tensor, eps: float = 1e-5) -> Tensor:
    """Performs RMS normalization on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w : Tensor
        Weight tensor for scaling the distribution.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-5``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.RMSNorm`
    """
    return RMSNormFunction.forward(PseudoContext(), x, w, eps)
