"""Neural network module base module"""

from __future__ import annotations
from abc import ABC
import pickle
from typing import Callable, Optional
from ..parameter import Parameter
from ...tensor import Tensor, ShapeError
from ...types import DeviceLike


__all__ = ["Module", "Passthrough", "save_module", "load_module"]


class Module(ABC):
    """Module base class."""

    def __init__(self) -> None:
        """Module base class."""
        self.y: Optional[Tensor] = None
        self.backward_fn: Optional[Callable[[Tensor], Optional[Tensor]]] = None
        self.__device: DeviceLike = "cpu"
        self.__retain_values: bool = False
        self.__training: bool = False
        self.__trainable: bool = True

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------------------

    @property
    def device(self) -> DeviceLike:
        """Device the module tensors are stored on."""
        return self.__device

    def to_device(self, device: DeviceLike) -> None:
        """Moves the module to a specified device.

        Parameters
        ----------
        device : DeviceLike
            Device to move the tensor to ("cuda" or "cpu").
        """
        if device == self.device:
            return
        self.__device = device

        if self.y is not None:
            self.y.to_device(device)

        for p in self.parameters:
            p.to_device(device)

    @property
    def retain_values(self) -> bool:
        """Whether to retain intermediate values after a forward pass."""
        return self.__retain_values

    def set_retain_values(self, value: bool) -> None:
        """Whether to retain intermediate values after a forward pass."""
        if self.__retain_values == value:
            return
        self.__retain_values = value

    @property
    def training(self) -> bool:
        """Training mode for the module."""
        return self.__training

    def set_training(self, value: bool) -> None:
        """Sets the training mode for the module"""
        if self.__training == value:
            return
        self.__training = value

    @property
    def trainable(self) -> bool:
        """Whether the module parameters are trainable."""
        return self.__trainable

    def set_trainable(self, value: bool) -> None:
        """Whether the module parameters are trainable."""
        if self.__trainable == value:
            return
        self.__trainable = value

        for parameter in self.parameters:
            parameter.requires_grad = value

    @property
    def parameters(self) -> list[Parameter]:
        """Returns the list of module parameters."""
        return [i[1] for i in self.__dict__.items() if isinstance(i[1], Parameter)]


    # ----------------------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, x: Tensor) -> Tensor:
        y = self.forward(x)
        self.set_y(y)
        return y

    # ----------------------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the module.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        ----------
        Tensor
            Computed module output.
        """
        return x

    def backward(self, dy: Tensor) -> Optional[Tensor]:
        """Performs a backward pass through the module.

        Parameters
        ----------
        dy : Tensor
            Output gradient tensor.

        Returns
        ----------
        Tensor, optional
            Input gradient tensor.
        """
        self.set_dy(dy)
        return dy if self.backward_fn is None else self.backward_fn(dy)

    def set_y(self, y: Tensor) -> None:
        """Saves the module output to y tensor.

        Parameters
        ----------
        y : Tensor
            Module output tensor.
        """
        if self.retain_values:
            self.y = y.copy()

    def set_dy(self, dy: Tensor) -> None:
        """Saves the module output gradients to y tensor.

        Parameters
        ----------
        dy : Tensor
            Module output tensor gradients.
        """
        if self.retain_values and self.y is not None:
            self.y.grad = dy.copy()

    def reset(self) -> None:
        """Resets temporary values like outputs and gradients."""
        self.y = None
        self.backward_fn = None

        for p in self.parameters:
            p.grad = None

    def check_dims(self, x: Tensor, valid_dims: list[int]) -> None:
        """Checks if a tensors dimensions match desired target dimensions.

        Parameters
        ----------
        x : Tensor
            Tensor whose dimensions are checked.
        valid_dims : int
            Valid numbers of dimension the tensor should have.

        Raises
        ------
        ShapeError
            If the tensor's dimensions do not match the target dimensions.
        """
        if x.ndim not in valid_dims:
            sender = self.__class__.__name__
            vdims = ", ".join([str(d) for d in valid_dims])
            raise ShapeError(
                f"{sender}: Number of input dimensions {
                    x.ndim} is not valid (valid: {vdims})"
            )

class Passthrough(Module):
    """Acts as a passthrough for data."""

def save_module(module: Module, filepath: str) -> None:
    """Saves a model as a binary file.

    Parameters
    ----------
    model : Model
        Model to be saved.
    filepath : str
        Path to the file.
    """

    module.to_device("cpu")
    module.reset()

    file = open(filepath, "wb")
    pickle.dump(module, file)
    file.close()


def load_module(filepath: str) -> Module:
    """Load a module from a previously saved binary file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    Model
        Loaded model.
    """
    file = open(filepath, "rb")
    obj = pickle.load(file)
    file.close()
    return obj
