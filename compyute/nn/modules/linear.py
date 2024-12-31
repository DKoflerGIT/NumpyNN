"""Neural network linear transformation modules."""

import math
from typing import Optional

from ...random import uniform
from ...tensors import Tensor
from ..functional.linear_funcs import LinearFunction
from ..parameter import Parameter
from .module import Module

__all__ = ["Linear"]


class Linear(Module):
    r"""Applies a linear transformation to the input.

    .. math::
        y = xW^T + b

    Shapes:
        - Input :math:`(B_1, ... , B_n, C_{in})`
        - Output :math:`(B_1, ... , B_n, C_{out})`
    where
        - :math:`B_1, ... , B_n` ... batch dimensions
        - :math:`C_{in}` ... input channels
        - :math:`C_{out}` ... output channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (neurons).
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{in}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # init parameters
        k = 1.0 / math.sqrt(in_channels)
        self.w = Parameter(uniform((out_channels, in_channels), -k, k))
        self.b = None if not bias else Parameter(uniform((out_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return LinearFunction.forward(self.function_ctx, x, self.w, self.b)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw, db = LinearFunction.backward(self.function_ctx, dy)
        self.update_parameter_grad(self.w, dw)
        self.update_parameter_grad(self.b, db)
        return dx
