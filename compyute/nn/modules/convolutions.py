"""Neural network convolution modules."""

import math
from typing import Literal, Optional

from ...random import uniform
from ...tensors import Tensor
from ..functional.convolution_funcs import (
    Conv1DFunction,
    Conv2DFunction,
    ConvTranspose1DFunction,
    ConvTranspose2DFunction,
)
from ..parameter import Parameter
from .module import Module

__all__ = ["Conv1D", "Conv2D", "ConvTranspose1D", "ConvTranspose2D"]


PaddingLike = int | Literal["valid", "same"]


def _str_to_pad(
    padding: Literal["valid", "same"], kernel_size: int, dilation: int
) -> int:
    if padding == "valid":
        return 0
    return (kernel_size * dilation - 1) // 2


class Conv1D(Module):
    r"""Applies a 1D convolution to the input for feature extraction.

    .. math::
        y = b + \sum_{k=0}^{C_{in}-1} w_{k}*x_{k}

    where :math:`*` is the cross-correlation operator.

    Shapes:
        - Input :math:`(B, C_{in}, S_{in})`
        - Output :math:`(B, C_{out}, S_{out})`
    where
        - :math:`B` ... batch dimension
        - :math:`C_{in}` ... input channels
        - :math:`S_{in}` ... input sequence
        - :math:`C_{out}` ... output channels
        - :math:`S_{out}` ... output sequence

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (filters).
    kernel_size : int
        Size of each kernel.
    padding : PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int, optional
        Stride used for the convolution operation. Defaults to ``1``.
    dilation : int, optional
        Dilation factor used for the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} \cdot \text{kernel_size}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (
            padding
            if isinstance(padding, int)
            else _str_to_pad(padding, kernel_size, dilation)
        )
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # init parameters
        k = 1.0 / math.sqrt(in_channels * kernel_size)
        w_shape = (out_channels, in_channels, kernel_size)
        self.w = Parameter(uniform(w_shape, -k, k))
        self.b = None if not bias else Parameter(uniform((out_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return Conv1DFunction.forward(
            self.ctx,
            x,
            self.w,
            self.b,
            self.padding,
            self.stride,
            self.dilation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = Conv1DFunction.backward(self.ctx, dy)
        self.update_parameter_grad(self.w, dw)
        self.update_parameter_grad(self.b, db)
        return dx


class Conv2D(Module):
    r"""Applies a 2D convolution to the input for feature extraction.

    .. math::
        y = b + \sum_{k=0}^{C_{in}-1} w_{k}*x_{k}

    where :math:`*` is the cross-correlation operator.

    Shapes:
        - Input :math:`(B, C_{in}, Y_{in}, X_{in})`
        - Output :math:`(B, C_{out}, Y_{out}, X_{out})`
    where
        - :math:`B` ... batch dimension
        - :math:`C_{in}` ... input channels
        - :math:`Y_{in}` ... input height
        - :math:`X_{in}` ... input width
        - :math:`C_{out}` ... output channels
        - :math:`Y_{out}` ... output height
        - :math:`X_{out}` ... output width

    Parameters
    ----------
    in_channels : int
        Number of input channels (color channels).
    out_channels : int
        Number of output channels (filters or feature maps).
    kernel_size : int
        Size of each kernel.
    padding : PaddingLike, optional
        Padding applied before convolution. Defaults to ``valid``.
    stride : int , optional
        Stride used for the convolution operation. Defaults to ``1``.
    dilation : int , optional
        Dilation factor used for the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * \text{kernel_size}^2}}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (
            padding
            if isinstance(padding, int)
            else _str_to_pad(padding, kernel_size, dilation)
        )
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # init parameters
        k = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.w = Parameter(uniform(w_shape, -k, k))
        self.b = None if not bias else Parameter(uniform((out_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return Conv2DFunction.forward(
            self.ctx,
            x,
            self.w,
            self.b,
            self.padding,
            self.stride,
            self.dilation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = Conv2DFunction.backward(self.ctx, dy)
        self.update_parameter_grad(self.w, dw)
        self.update_parameter_grad(self.b, db)
        return dx


class ConvTranspose1D(Module):
    r"""Applies a 1D transposed convolution to the input for upsampling as descirbed by
    `Long et al., 2015 <https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf>`_.

    Shapes:
        - Input :math:`(B, C_{in}, S_{in})`
        - Output :math:`(B, C_{out}, S_{out})`
    where
        - :math:`B` ... batch dimension
        - :math:`C_{in}` ... input channels
        - :math:`S_{in}` ... input sequence
        - :math:`C_{out}` ... output channels
        - :math:`S_{out}` ... output sequence

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (filters).
    kernel_size : int
        Size of each kernel.
    padding : PaddingLike, optional
        Number of cols removed from the output. Defaults to ``valid``.
    stride : int , optional
        Stride used for the deconvolution operation. Defaults to ``1``.
    dilation : int , optional
        Dilation factor used for the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * \text{kernel_size}^2}}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (
            padding
            if isinstance(padding, int)
            else _str_to_pad(padding, kernel_size, dilation)
        )
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # init parameters
        k = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.w = Parameter(uniform(w_shape, -k, k))
        self.b = None if not bias else Parameter(uniform((out_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return ConvTranspose1DFunction.forward(
            self.ctx,
            x,
            self.w,
            self.b,
            self.padding,
            self.stride,
            self.dilation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = ConvTranspose1DFunction.backward(self.ctx, dy)
        self.update_parameter_grad(self.w, dw)
        self.update_parameter_grad(self.b, db)
        return dx


class ConvTranspose2D(Module):
    r"""Applies a 2D transposed convolution to the input for upsampling as descirbed by
    `Long et al., 2015 <https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf>`_.

    Shapes:
        - Input :math:`(B, C_{in}, Y_{in}, X_{in})`
        - Output :math:`(B, C_{out}, Y_{out}, X_{out})`
    where
        - :math:`B` ... batch dimension
        - :math:`C_{in}` ... input channels
        - :math:`Y_{in}` ... input height
        - :math:`X_{in}` ... input width
        - :math:`C_{out}` ... output channels
        - :math:`Y_{out}` ... output height
        - :math:`X_{out}` ... output width

    Parameters
    ----------
    in_channels : int
        Number of input channels (color channels).
    out_channels : int
        Number of output channels (filters or feature maps).
    kernel_size : int
        Size of each kernel.
    padding : PaddingLike, optional
        Number of rows and cols removed from the output. Defaults to ``valid``.
    stride : int , optional
        Stride used for the deconvolution operation. Defaults to ``1``.
    dilation : int , optional
        Dilation factor used for the filter. Defaults to ``1``.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in} * \text{kernel_size}^2}}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: PaddingLike = "valid",
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (
            padding
            if isinstance(padding, int)
            else _str_to_pad(padding, kernel_size, dilation)
        )
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # init parameters
        k = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.w = Parameter(uniform(w_shape, -k, k))
        self.b = None if not bias else Parameter(uniform((out_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return ConvTranspose2DFunction.forward(
            self.ctx,
            x,
            self.w,
            self.b,
            self.padding,
            self.stride,
            self.dilation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = ConvTranspose2DFunction.backward(self.ctx, dy)
        self.update_parameter_grad(self.w, dw)
        self.update_parameter_grad(self.b, db)
        return dx
