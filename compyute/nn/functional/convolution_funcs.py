"""Neural network convolution functions."""

from typing import Optional

from ...tensor_ops.creation_ops import zeros
from ...tensor_ops.multiary_ops import einsum
from ...tensor_ops.shape_ops import flip, pad, pad_to_shape, pooling1d, pooling2d
from ...tensors import ShapeError, Tensor
from .functions import Function, FunctionContext, PseudoContext

__all__ = [
    "conv1d",
    "dilate1d",
    "pad1d",
    "conv2d",
    "dilate2d",
    "pad2d",
    "conv_transpose1d",
    "conv_transpose2d",
]


class Conv1DFunction(Function):
    """Computes the convolution of two tensors over their last dimension."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        f: Tensor,
        b: Optional[Tensor],
        padding: int,
        stride: int,
        dilation: int,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")

        f = Dilation1DFunction.forward(ctx, f, dilation)
        x = Pad1DFunction.forward(ctx, x, padding)
        y = RawConv1DFunction.forward(ctx, x, f, stride)
        if b:
            y += b.view((*b.shape, 1))

        ctx.add(b is not None)
        return y

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        b = ctx.get()

        dx, df = RawConv1DFunction.backward(ctx, dy)
        dx = Pad1DFunction.backward(ctx, dx)
        df = Dilation1DFunction.backward(ctx, df)
        db = None if not b else dy.sum((0, 2))

        return dx, df, db


def conv1d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the convolution of two tensors over their last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None``. If ``None``, no bias is added.
    padding : int, optional
        Padding applied to the input tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Convolution1D`
    """
    return Conv1DFunction.forward(PseudoContext(), x, f, b, padding, stride, dilation)


class Dilation1DFunction(Function):
    """Dilates a tensor in its last dimension."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, dilation: int) -> Tensor:
        no_dilation = dilation == 1
        ctx.add(no_dilation, dilation)
        if no_dilation:
            return x

        y_shape = (*x.shape[:-1], dilation * (x.shape[-1] - 1) + 1)
        y = zeros(y_shape, device=x.device, dtype=x.dtype)
        y[..., ::dilation] = x

        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        no_dilation, dilation = ctx.get()
        if no_dilation:
            return dy
        return dy[..., ::dilation]


def dilate1d(x: Tensor, dilation: int) -> Tensor:
    """Dilates a tensor in its last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : int
        Dilation factor to use.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return Dilation1DFunction.forward(PseudoContext(), x, dilation)


class Pad1DFunction(Function):
    """Pads a tensor in its last dimension."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        ctx.add(no_padding, padding)
        if no_padding:
            return x

        widths = tuple([(0, 0)] * (x.ndim - 1) + [(padding, padding)])
        y = pad(x, widths)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        no_padding, padding = ctx.get()
        if no_padding:
            return dy
        return dy[..., padding:-padding].to_contiguous()


def pad1d(x: Tensor, padding: int) -> Tensor:
    """Pads a tensor in its last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : int
        Padding width applied to the beginning and end of the last dimension.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return Pad1DFunction.forward(PseudoContext(), x, padding)


class RawConv1DFunction(Function):
    """Computes the 1D convolution of two tensors."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, f: Tensor, stride: int) -> Tensor:
        x_pooled = pooling1d(x, f.shape[-1], stride)  # view as (B, Ci, So, F)
        y = einsum("bitf,oif->bot", x_pooled, f).to_contiguous()  # multiply and add
        ctx.add(x, f, stride)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> tuple[Tensor, Tensor]:
        x, f, stride = ctx.get()

        # fill elements skipped by strides with zeros
        dy = dilate1d(dy, stride)

        # pad to match unstrided dy
        dy_t = x.shape[-1] - f.shape[-1] + 1
        dy = pad_to_shape(dy, (*dy.shape[:-1], dy_t))

        # full pad
        dy = pad1d(dy, f.shape[-1] - 1)

        # input grads
        dy_pooled = pooling1d(dy, f.shape[-1])  # view as (B, Co, Si, F)
        f = flip(f, dim=-1)
        dx = einsum("bosf,oif->bis", dy_pooled, f).to_contiguous()

        # filter grads
        dy_pooled = pooling1d(dy, x.shape[-1])  # view as (B, Co, F, Si)
        df = einsum("bofs,bis->oif", dy_pooled, x)
        df = flip(df, dim=-1).to_contiguous()

        return dx, df


class Conv2DFunction(Function):
    """Computes the convolution of two tensors over their last dimension."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        f: Tensor,
        b: Optional[Tensor],
        padding: int,
        stride: int,
        dilation: int,
    ) -> Tensor:
        if x.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")

        f = Dilation2DFunction.forward(ctx, f, dilation)
        x = Pad2DFunction.forward(ctx, x, padding)
        y = RawConv2DFunction.forward(ctx, x, f, stride)
        if b:
            y += b.view((*b.shape, 1, 1))

        ctx.add(b is not None)
        return y

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        b = ctx.get()

        dx, df = RawConv2DFunction.backward(ctx, dy)
        dx = Pad2DFunction.backward(ctx, dx)
        df = Dilation2DFunction.backward(ctx, df)
        db = None if not b else dy.sum((0, 2, 3))

        return dx, df, db


def conv2d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the convolution of two tensors over their last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None. If ``None``, no bias is added.
    padding : int, optional
        Padding applied to the input tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Convolution2D`
    """
    return Conv2DFunction.forward(PseudoContext(), x, f, b, padding, stride, dilation)


class Dilation2DFunction(Function):
    """Dilates a tensor in its last two dimensions."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, dilation: int) -> Tensor:
        no_dilation = dilation == 1
        ctx.add(no_dilation, dilation)
        if no_dilation:
            return x

        y_height = dilation * (x.shape[-2] - 1) + 1
        y_width = dilation * (x.shape[-1] - 1) + 1
        y = zeros((*x.shape[:-2], y_height, y_width), device=x.device, dtype=x.dtype)
        y[..., ::dilation, ::dilation] = x

        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        no_dilation, dilation = ctx.get()
        if no_dilation:
            return dy
        return dy[..., ::dilation, ::dilation]


def dilate2d(x: Tensor, dilation: int) -> Tensor:
    """Dilates a tensor in its last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dilation : int
        Dilation factor to use.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return Dilation2DFunction.forward(PseudoContext(), x, dilation)


class Pad2DFunction(Function):
    """Pads a tensor in its last two dimensions."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        ctx.add(no_padding, padding)
        if no_padding:
            return x
        widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
        y = pad(x, widths)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        no_padding, padding = ctx.get()
        if no_padding:
            return dy
        return dy[..., padding:-padding, padding:-padding].to_contiguous()


def pad2d(x: Tensor, padding: int) -> Tensor:
    """Pads a tensor in its last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    padding : int
        Padding width applied to the beginning and end of the last two dimensions.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return Pad2DFunction.forward(PseudoContext(), x, padding)


class RawConv2DFunction(Function):
    """Computes the 2D convolution of two tensors."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, f: Tensor, stride: int) -> Tensor:
        x_pooled = pooling2d(x, f.shape[-1], stride)  # view as (B, Ci, Y, X, Fy, Fx)
        y = einsum("biyxjk,oijk->boyx", x_pooled, f).to_contiguous()  # multiply and add
        ctx.add(x, f, stride)
        return y

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> tuple[Tensor, Tensor]:
        x, f, stride = ctx.get()

        # fill elements skipped by strides with zeros
        dy = dilate2d(dy, stride)

        # pad to match unstrided dy
        dy_t = x.shape[-1] - f.shape[-1] + 1
        dy = pad_to_shape(dy, (*dy.shape[:-2], dy_t, dy_t))

        # full pad
        dy = pad2d(dy, f.shape[-1] - 1)

        # input grads
        dy_pooled = pooling2d(dy, f.shape[-1])  # view as (B, Co, Y, X, Fy, Fx)
        f = flip(f, dim=(-2, -1))
        dx = einsum("boyxjk,oijk->biyx", dy_pooled, f).to_contiguous()

        # filter grads
        dy_pooled = pooling2d(dy, x.shape[-1])  # view as (B, Co, Fy, Fx, Y, X)
        df = einsum("bojkyx,biyx->oijk", dy_pooled, x)
        df = flip(df, dim=(-2, -1)).to_contiguous()

        return dx, df


class InvPad1DFunction(Function):
    """Removes cols from a tensor's last dimension."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        ctx.add(no_padding, padding)
        if no_padding:
            return x
        return x[..., padding:-padding].to_contiguous()

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        no_padding, padding = ctx.get()
        if no_padding:
            return dy
        widths = tuple([(0, 0)] * (dy.ndim - 1) + [(padding, padding)])
        return pad(dy, widths)


class ConvTranspose1DFunction(Function):
    """Computes the transposed convolution of two tensors over their last dimension."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        f: Tensor,
        b: Optional[Tensor],
        padding: int,
        stride: int,
        dilation: int,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")

        f = flip(f, -1)
        f = Dilation1DFunction.forward(ctx, f, dilation)
        x = Dilation1DFunction.forward(ctx, x, stride)
        x = Pad1DFunction.forward(ctx, x, f.shape[-1] - 1)  # full pad
        y = RawConv1DFunction.forward(ctx, x, f, stride=1)
        y = InvPad1DFunction.forward(ctx, y, padding)
        if b:
            y += b.view((*b.shape, 1))

        ctx.add(b is not None)
        return y

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        b = ctx.get()

        dy = InvPad1DFunction.backward(ctx, dy)
        dx, df = RawConv1DFunction.backward(ctx, dy)
        dx = Pad1DFunction.backward(ctx, dx)
        dx = Dilation1DFunction.backward(ctx, dx)
        df = Dilation1DFunction.backward(ctx, df)
        df = flip(df, -1).to_contiguous()
        db = None if not b else dy.sum((0, 2))

        return dx, df, db


def conv_transpose1d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the transposed convolution of two tensors over their last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None. If ``None``, no bias is added.
    padding : int, optional
        Number of rows and cols removed from the output tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the deconvolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.ConvTranspose1D`
    """
    return ConvTranspose1DFunction.forward(
        PseudoContext(), x, f, b, padding, stride, dilation
    )


class InvPad2DFunction(Function):
    """Removes rows and cols from a tensor's last two dimensions."""

    @staticmethod
    def forward(ctx: FunctionContext, x: Tensor, padding: int) -> Tensor:
        no_padding = padding == 0
        ctx.add(no_padding, padding)
        if no_padding:
            return x
        return x[..., padding:-padding, padding:-padding].to_contiguous()

    @staticmethod
    def backward(ctx: FunctionContext, dy: Tensor) -> Tensor:
        no_padding, padding = ctx.get()
        if no_padding:
            return dy
        widths = tuple([(0, 0)] * (dy.ndim - 2) + [(padding, padding)] * 2)
        return pad(dy, widths)


class ConvTranspose2DFunction(Function):
    """Computes the transposed convolution of two tensors over their last two dimensions."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        f: Tensor,
        b: Optional[Tensor],
        padding: int,
        stride: int,
        dilation: int,
    ) -> Tensor:
        if x.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {x.ndim}D.")

        f = flip(f, (-2, -1))
        f = Dilation2DFunction.forward(ctx, f, dilation)
        x = Dilation2DFunction.forward(ctx, x, stride)
        x = Pad2DFunction.forward(ctx, x, f.shape[-1] - 1)  # full pad
        y = RawConv2DFunction.forward(ctx, x, f, stride=1)
        y = InvPad2DFunction.forward(ctx, y, padding)
        if b:
            y += b.view((*b.shape, 1, 1))

        ctx.add(b is not None)
        return y

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        b = ctx.get()

        dy = InvPad2DFunction.backward(ctx, dy)
        dx, df = RawConv2DFunction.backward(ctx, dy)
        dx = Pad2DFunction.backward(ctx, dx)
        dx = Dilation2DFunction.backward(ctx, dx)
        df = Dilation2DFunction.backward(ctx, df)
        df = flip(df, (-2, -1)).to_contiguous()
        db = None if not b else dy.sum((0, 2, 3))

        return dx, df, db


def conv_transpose2d(
    x: Tensor,
    f: Tensor,
    b: Optional[Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
) -> Tensor:
    """Computes the transposed convolution of two tensors over their last two dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    f : Tensor
        Filter tensor.
    b : Tensor, optional
        Bias tensor. Defaults to ``None. If ``None``, no bias is added.
    padding : int, optional
        Number of rows and cols removed from the output tensor. Defaults to ``0``.
    stride : int, optional
        Stride used in the deconvolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor used for the filter. Defaults to ``1``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Deconvolution2D`
    """
    return ConvTranspose2DFunction.forward(
        PseudoContext(), x, f, b, padding, stride, dilation
    )
