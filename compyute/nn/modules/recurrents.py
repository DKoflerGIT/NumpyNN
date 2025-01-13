"""Neural network recurrent modules."""

import math
from typing import Literal, Optional

from ...random import uniform
from ...tensors import Tensor
from ..functional.recurrent_funcs import GRUFunction, LSTMFunction, RecurrentFunction
from ..parameter import Parameter
from .module import Module

__all__ = ["GRU", "LSTM", "Recurrent"]


class Recurrent(Module):
    r"""Elman Recurrent module.
    For each element in the sequence the hidden state is computed as follows:

    .. math::
        h_t = tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

    .. note::
        This implementation follows the one by PyTorch which differs slightly from the original paper.

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{h})`
    where
        - :math:`B` ... batch dimension
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`C_{h}` ... hidden channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of hidden channels.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: str = "tanh",
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.activation = activation

        # init parameters
        k = 1.0 / math.sqrt(h_channels)
        self.w_i = Parameter(uniform((h_channels, in_channels), -k, k))
        self.b_i = Parameter(uniform((h_channels,), -k, k))
        self.w_h = Parameter(uniform((h_channels, h_channels), -k, k))
        self.b_h = Parameter(uniform((h_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return RecurrentFunction.forward(
            self.ctx,
            x,
            self.w_i,
            self.b_i,
            self.w_h,
            self.b_h,
            self.activation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw_i, db_i, dw_h, db_h = RecurrentFunction.backward(self.ctx, dy)
        self.apply_grads(
            (self.w_i, self.b_i, self.w_h, self.b_h), (dw_i, db_i, dw_h, db_h)
        )
        return dx


class LSTM(Module):
    r"""Long Short-Term Memory module as described by
    `Hochreiter et al., 1997 <https://www.bioinf.jku.at/publications/older/2604.pdf>`_.
    For each element in the sequence the hidden state is computed as follows:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(x_tW_{ii}^T + b_{ii} + h_{t-1}W_{hi}^T + b_{hi}) \\
            f_t = \sigma(x_tW_{if}^T + b_{if} + h_{t-1}W_{hf}^T + b_{hf}) \\
            g_t = act(x_tW_{ig}^T + b_{ig} + h_{t-1}W_{hg}^T + b_{hg}) \\
            o_t = \sigma(x_tW_{io}^T + b_{io} + h_{t-1}W_{ho}^T + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot act(c_t) \\
        \end{array}

    where :math:`\odot` is the Hadamard product.

    .. note::
        This implementation follows the one by PyTorch which differs slightly from the original paper.

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{h})`
    where
        - :math:`B` ... batch dimension
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`C_{h}` ... hidden channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of hidden channels.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.activation = activation

        # init parameters
        k = 1.0 / math.sqrt(h_channels)
        self.w_i = Parameter(uniform((4 * h_channels, in_channels), -k, k))
        self.b_i = None if not bias else Parameter(uniform((4 * h_channels,), -k, k))
        self.w_h = Parameter(uniform((4 * h_channels, h_channels), -k, k))
        self.b_h = None if not bias else Parameter(uniform((4 * h_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return LSTMFunction.forward(
            self.ctx,
            x,
            self.w_i,
            self.b_i,
            self.w_h,
            self.b_h,
            self.activation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw_i, db_i, dw_h, db_h = LSTMFunction.backward(self.ctx, dy)
        self.apply_grads(
            (self.w_i, self.b_i, self.w_h, self.b_h), (dw_i, db_i, dw_h, db_h)
        )
        return dx


class GRU(Module):
    r"""Gated Recurrent Unit module as described by
    `Cho et al., 2014 <https://arxiv.org/pdf/1406.1078>`_.
    For each element in the sequence the hidden state is computed as follows:

    .. math::
        \begin{array}{ll} \\
            r_t = \sigma(x_tW_{ir}^T + b_{ir} + h_{t-1}W_{hr}^T + b_{hr}) \\
            z_t = \sigma(x_tW_{iz}^T + b_{iz} + h_{t-1}W_{hz}^T + b_{hz}) \\
            n_t = act(x_tW_{in}^T + b_{in} + r_t \odot (h_{t-1}W_{hn}^T + b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
        \end{array}

    where :math:`\odot` is the Hadamard product.
    
    .. note::
        This implementation follows the one by PyTorch which differs slightly from the original paper.

    Shapes:
        - Input :math:`(B, S, C_{in})`
        - Output :math:`(B, S, C_{h})`
    where
        - :math:`B` ... batch dimension
        - :math:`S` ... sequence
        - :math:`C_{in}` ... input channels
        - :math:`C_{h}` ... hidden channels

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    h_channels : int
        Number of hidden channels.
    bias : bool, optional
        Whether to use bias values. Defaults to ``True``.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.


    .. note::
        Weights and biases are initialized from :math:`\mathcal{U}(-k, k)`, where
        :math:`k = \sqrt{\frac{1}{C_{h}}}`.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int,
        bias: bool = True,
        activation: Literal["relu", "tanh"] = "tanh",
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label)
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.bias = bias
        self.activation = activation

        # init parameters
        k = 1.0 / math.sqrt(h_channels)
        self.w_i = Parameter(uniform((h_channels, in_channels), -k, k))
        self.b_i = None if not bias else Parameter(uniform((h_channels,), -k, k))
        self.w_h = Parameter(uniform((h_channels, h_channels), -k, k))
        self.b_h = None if not bias else Parameter(uniform((h_channels,), -k, k))

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return GRUFunction.forward(
            self.ctx,
            x,
            self.w_i,
            self.b_i,
            self.w_h,
            self.b_h,
            self.activation,
        )

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx, dw_i, db_i, dw_h, db_h = GRUFunction.backward(self.ctx, dy)
        self.apply_grads(
            (self.w_i, self.b_i, self.w_h, self.b_h), (dw_i, db_i, dw_h, db_h)
        )
        return dx
