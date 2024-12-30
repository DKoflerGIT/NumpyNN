"""Loss functions."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Literal

from ..tensor_ops.unary_ops import is_nan
from ..tensors import Tensor
from .functional.functions import FunctionContext
from .functional.loss_funcs import (
    BCELossFunction,
    CrossEntropyLossFunction,
    DiceLossFunction,
    MSELossFunction,
)

__all__ = ["Loss", "BCELoss", "CrossEntropyLoss", "MSELoss", "DiceLoss"]


DEBUG = bool(os.environ.get("COMPYUTE_DEBUG", False))


class Loss(ABC):
    """Loss base class."""

    def __init__(self) -> None:
        self.function_ctx = FunctionContext()
        self.label = self.__class__.__name__
        self._is_training = True

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)

    @abstractmethod
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Computes the loss given model predictions and target values.

        Parameters
        ----------
        logits : Tensor
            Model logits.
        targets : Tensor
            Target values.

        Returns
        -------
        Tensor
            Computed loss value.
        """

    @abstractmethod
    def backward(self) -> Tensor:
        """Computes the gradient of the loss with respect to the model's predictions.

        Returns
        -------
        Tensor
            The loss gradient.
        """

    @staticmethod
    def register_forward(fwd_fn: Callable) -> Callable:
        """Registers a the forward method to the loss function."""

        @wraps(fwd_fn)
        def wrapper(l: Loss, logits: Tensor, targets: Tensor) -> Tensor:
            l.function_ctx.context.clear()

            if DEBUG:
                dt = time.perf_counter()
                loss = fwd_fn(l, logits, targets)
                dt = (time.perf_counter() - dt) * 1e3
                print(
                    f"{l.label:20s} | fwd | "
                    f"{logits.dtype:15s} | "
                    f"{targets.dtype:15s} | "
                    f"{loss.dtype:15s} | "
                    f"{dt=:>10.4f} ms"
                )
            else:
                loss = fwd_fn(l, logits, targets)

            assert not is_nan(loss).any().item(), "NaNs detected in " + repr(l)
            return loss

        return wrapper

    @staticmethod
    def register_backward(bwd_fn: Callable) -> Callable:
        """Registers a the backward method to the loss function."""

        @wraps(bwd_fn)
        def wrapper(l: Loss) -> Tensor:
            if DEBUG:
                dt = time.perf_counter()
                dx = bwd_fn(l)
                dt = (time.perf_counter() - dt) * 1e3
                print(f"{l.label:20s} | bwd | {dx.dtype:15s} | {dt=:>10.4f} ms")
            else:
                dx = bwd_fn(l)

            assert not is_nan(dx).any().item(), "NaNs detected in " + repr(l)
            assert not l.function_ctx.context, "Context not cleared in " + repr(l)
            return dx

        return wrapper


class MSELoss(Loss):
    r"""Computes the mean squared error loss.

    .. math::
        L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2

    where
        - :math:`\hat{y}` ... model logits
        - :math:`y` ... ground truth
    """

    @Loss.register_forward
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return MSELossFunction.forward(self.function_ctx, logits, targets)

    @Loss.register_backward
    def backward(self) -> Tensor:
        return MSELossFunction.backward(self.function_ctx)


class CrossEntropyLoss(Loss):
    r"""Computes the cross entropy loss from logits.

    .. math::
        L = -\frac{1}{N} \sum_{i=1}^N y_i \cdot \log(\text{softmax}(\hat{y}_i))

    where
        - :math:`\hat{y}` ... model logits
        - :math:`y` ... ground truth
    """

    @Loss.register_forward
    def forward(self, logits: Tensor, targets: Tensor, eta: float = 1e-8) -> Tensor:
        return CrossEntropyLossFunction.forward(self.function_ctx, logits, targets, eta)

    @Loss.register_backward
    def backward(self) -> Tensor:
        return CrossEntropyLossFunction.backward(self.function_ctx)


class BCELoss(Loss):
    r"""Computes the binary cross entropy loss from logits.

    .. math::
        L = -\frac{1}{N} \sum_{i=1}^N y_i \cdot \log(\hat{y}_i) - (1 - y_i) \cdot \log(1 - \hat{y}_i)

    The above version can be numerically instable, therefore this equivalent formulation is used:

    .. math::
        L = -\frac{1}{N} \sum_{i=1}^N \text{max}(0,\hat{y}_i) - \hat{y}_i \cdot y_i + \text{log}(1+e^{-|y_i|})

    where
        - :math:`\hat{y}` ... model logits
        - :math:`y` ... ground truth
    """

    @Loss.register_forward
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return BCELossFunction.forward(self.function_ctx, logits, targets)

    @Loss.register_backward
    def backward(self) -> Tensor:
        return BCELossFunction.backward(self.function_ctx)


class DiceLoss(Loss):
    r"""Computes the dice loss from logits as described by
    `Milletari et al., 2016 <https://arxiv.org/pdf/1606.04797>`_.

    .. math::
        L = 1 - \frac{1}{C} \sum_{c=0}^{C-1} \frac{2 \sum_{n=1}^N \hat{y}_n^cy_n^c}{\sum_{n=1}^N \hat{y}_n^c + y_n^c}

    where
        - :math:`C` ... number of classes
        - :math:`\hat{y}` ... model logits
        - :math:`y` ... ground truth
    """

    @Loss.register_forward
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return DiceLossFunction.forward(self.function_ctx, logits, targets)

    @Loss.register_backward
    def backward(self) -> Tensor:
        return DiceLossFunction.backward(self.function_ctx)


LossLike = Loss | Literal["bce", "cross_entropy", "mse", "dice"]
LOSSES: dict[str, type[Loss]] = {
    "bce": BCELoss,
    "cross_entropy": CrossEntropyLoss,
    "mse": MSELoss,
    "dice": DiceLoss,
}


def get_loss_function(loss: LossLike) -> Loss:
    """Returns an instance of a loss function."""
    if isinstance(loss, Loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(f"Unknown loss function: {loss}.")
    return LOSSES[loss]()
