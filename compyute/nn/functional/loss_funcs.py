"""Neural network loss functions."""

import math

from ...preprocessing.basic import one_hot_encode
from ...tensor_ops.selection_ops import maximum
from ...tensor_ops.unary_ops import abs, exp, log
from ...tensors import ShapeError, Tensor
from .activation_funcs import SoftmaxFunction, sigmoid, softmax
from .functions import Function, FunctionContext, PseudoContext

__all__ = ["mse_loss", "cross_entropy_loss", "bce_loss", "dice_loss"]


class MSELossFunction(Function):
    """Computes the mean squared error loss."""

    @staticmethod
    def forward(ctx: FunctionContext, logits: Tensor, targets: Tensor) -> Tensor:
        diff = logits - targets
        loss = (diff * diff).mean()
        ctx.add(logits.size, diff)
        return loss

    @staticmethod
    def backward(ctx: FunctionContext) -> Tensor:
        logits_size, diff = ctx.get()
        return 2.0 * diff / float(logits_size)


def mse_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean squared error loss.

    Parameters
    ----------
    logits : Tensor
        Model logits.
    targets : Tensor
        Target values.

    Returns
    -------
    Tensor
        Mean squared error loss.

    See Also
    --------
    :class:`compyute.nn.MSELoss`
    """
    return MSELossFunction.forward(PseudoContext(), logits, targets)


class CrossEntropyLossFunction(Function):
    """Computes the cross entropy loss from logits."""

    @staticmethod
    def forward(
        ctx: FunctionContext, logits: Tensor, targets: Tensor, eta: float
    ) -> Tensor:
        probs = softmax(logits)
        targets = one_hot_encode(targets, logits.shape[-1], probs.dtype)
        loss = -(log(probs + eta) * targets).sum(-1).mean()
        ctx.add(targets, probs)
        return loss

    @staticmethod
    def backward(ctx: FunctionContext) -> Tensor:
        targets, probs = ctx.get()
        return (probs - targets) / float(math.prod(targets.shape[:-1]))


def cross_entropy_loss(logits: Tensor, targets: Tensor, eta: float = 1e-8) -> Tensor:
    """Computes the cross entropy loss from logits.

    Parameters
    ----------
    logits : Tensor
        Model logits.
    targets : Tensor
        Target class labels, must be of type ``int``.
    eta : float, optional
        A small constant added for numerical stability. Defaults to ``1e-8``.


    Returns
    -------
    Tensor
        Cross entropy loss.

    See Also
    --------
    :class:`compyute.nn.CrossEntropyLoss`
    """
    return CrossEntropyLossFunction.forward(PseudoContext(), logits, targets, eta)


class BCELossFunction(Function):
    """Computes the binary cross entropy loss from logits."""

    @staticmethod
    def forward(ctx: FunctionContext, logits: Tensor, targets: Tensor) -> Tensor:
        max_logits = maximum(logits, 0.0)
        loss = (max_logits - logits * targets + log(1 + exp(-abs(logits)))).mean()
        ctx.add(logits, targets)
        return loss

    @staticmethod
    def backward(ctx: FunctionContext) -> Tensor:
        logits, targets = ctx.get()
        return (sigmoid(logits) - targets) / float(logits.size)  # thank you ChatGPT


def bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Computes the binary cross entropy loss from logits.

    Parameters
    ----------
    logits : Tensor
        Model logits.
    targets : Tensor
        Binary target class labels, must be either ``0`` or ``1``.

    Returns
    -------
    Tensor
        Binary cross entropy loss.

    See Also
    --------
    :class:`compyute.nn.BCELoss`
    """
    return BCELossFunction.forward(PseudoContext(), logits, targets)


class DiceLossFunction(Function):
    """Computes the dice loss from logits."""

    @staticmethod
    def forward(
        ctx: FunctionContext, logits: Tensor, targets: Tensor, eps: float = 1e-5
    ) -> Tensor:
        if logits.ndim != 4:
            raise ShapeError(f"Expected input to be 4D, got {logits.ndim}D.")
        if targets.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {targets.ndim}D.")

        logits_shape = logits.shape
        logits = logits.view((*logits.shape[:2], -1))
        targets = targets.view((logits.shape[0], -1))

        # softmax along channel dim
        probs = SoftmaxFunction.forward(ctx, logits, dim=1)

        # one hot along channel dim
        targets = one_hot_encode(targets, logits.shape[1], probs.dtype).T

        intersection = (probs * targets).sum(dim=-1, keepdims=True)
        union = (probs * probs + targets * targets).sum(-1, keepdims=True)
        dice_coeff = (2 * intersection + eps) / (union + eps)
        loss = 1 - dice_coeff.mean()

        ctx.add(logits_shape, probs, targets, intersection + eps, union + eps)
        return loss

    @staticmethod
    def backward(ctx: FunctionContext) -> Tensor:
        logits_shape, probs, targets, intersection_eps, union_eps = ctx.get()
        ddice_coeff = -2 / float(math.prod(targets.shape[:2]))
        dintersection = ddice_coeff / union_eps
        dunion = -ddice_coeff * intersection_eps / union_eps**2
        dprobs = dintersection * targets + dunion * 2 * probs
        dlogits = SoftmaxFunction.backward(ctx, dprobs)
        return dlogits.view(logits_shape)


def dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Computes the dice loss from logits.

    Parameters
    ----------
    logits : Tensor
        Model logits.
    targets : Tensor
        Target class labels.

    Returns
    -------
    Tensor
        Dice loss.

    See Also
    --------
    :class:`compyute.nn.DiceLoss`
    """
    return DiceLossFunction.forward(PseudoContext(), logits, targets)
