"""Base tensor class."""

from __future__ import annotations

from typing import Any, Optional, TypeAlias

import numpy

from .backend import (
    ArrayLike,
    Device,
    cpu,
    cuda,
    data_to_device,
    get_default_device,
    get_device_from_array,
)
from .typing import (
    DTYPES,
    DType,
    ScalarLike,
    complex64,
    float16,
    float32,
    float64,
    get_default_dtype,
    int32,
    int64,
)
from .utils import get_debug_mode

__all__ = ["tensor", "Tensor", "newaxis"]

ShapeLike: TypeAlias = tuple[int, ...]
DimLike: TypeAlias = int | tuple[int, ...]


class ShapeError(Exception):
    """Incompatible tensor shapes."""


newaxis: TypeAlias = None


def tensor(
    data: ArrayLike | ScalarLike | list,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    """Creates a tensor object from arbitrary data.
    The data type and device are inferred from the data if not specified.

    Parameters
    ----------
    data : ArrayLike | ScalarLike | list
        Data to initialize the tensor data.
        Can be a list, tuple, NumPy/Cupy ndarray, scalar, and other types.
    device : Device, optional
        Device the tensor should be stored on. If ``None``, it is inferred from the data.
    dtype : DType, optional
        Data type of tensor data. If ``None``, it is inferred from the data.

    Returns
    -------
    Tensor
        Tensor object.
    """
    device = device or get_default_device() or get_device_from_array(data)
    dtype = dtype or get_default_dtype()
    array_dtype = dtype.t if dtype is not None else None
    with device:
        data = device.module.asarray(data, array_dtype)
    return Tensor(data)


class Tensor:
    """Tensor object used for storing multidimensional data.

    .. note::
        Tensors can only be initialized with NumPy or CuPy arrays.
        For other data types use the :func:`compyute.tensor` function. It automatically
        infers the data type and device if not specified.

    Parameters
    ----------
    data : ArrayLike
        Data to initialize the tensor. Must be a NumPy array or CuPy array.
    """

    __slots__ = "data", "grad", "_iterator"

    def __init__(self, data: ArrayLike) -> None:
        self.data = data
        self.grad: Optional[Tensor] = None
        self._iterator = 0

    # ----------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------
    @property
    def device(self) -> Device:
        """Device the tensor data is stored on."""
        return get_device_from_array(self.data)

    @property
    def dtype(self) -> DType:
        """Tensor data type."""
        return DTYPES[self.data.dtype.name]

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Tensor size (number of elements)."""
        return self.data.size

    @property
    def shape(self) -> ShapeLike:
        """Tensor shape."""
        return self.data.shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Tensor strides."""
        return self.data.strides

    @property
    def T(self) -> Tensor:
        """View of the tensor with its last two dimensions transposed."""
        if self.ndim < 2:
            return Tensor(self.data)
        return self.transpose(-1, -2)

    @property
    def ptr(self) -> int:
        """Pointer to the tensor data in memory."""
        return id(self.data)

    @property
    def contiguous(self) -> Optional[bool]:
        """Whether the tensor data is stored as a C-contiguous array."""
        return self.data.flags.c_contiguous

    @property
    def nbytes(self) -> int:
        """Number of bytes the tensor data occupies."""
        return self.data.nbytes

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__name__}("
        suffix = (
            ")"
            if not get_debug_mode()
            else f", shape={self.shape}, dtype={self.dtype.name}, device={self.device.name}, contiguous={self.contiguous})"
        )
        return (
            prefix
            + self.device.module.array2string(
                self.data,
                max_line_width=80,
                precision=4,
                separator=", ",
                prefix=prefix,
                floatmode="maxprec_equal",
            )
            + suffix
        )

    def __getitem__(self, key: Any) -> Tensor:
        if isinstance(key, tuple):
            return Tensor(self.data[key])
        key = to_arraylike(key)
        return Tensor(self.data[key])

    def __setitem__(self, key: Any, value: Tensor | ScalarLike) -> None:
        self.data[to_arraylike(key)] = to_arraylike(value)

    def __iter__(self) -> Tensor:
        self._iterator = 0
        return self

    def __next__(self) -> Tensor:
        if self._iterator == self.shape[0]:
            raise StopIteration
        y = self[self._iterator]
        self._iterator += 1
        return y

    def __add__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data + to_arraylike(other))

    def __radd__(self, other: Optional[ScalarLike]) -> Tensor:
        return Tensor(self.data + other)

    def __iadd__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data += to_arraylike(other)
        return self

    def __sub__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data - to_arraylike(other))

    def __rsub__(self, other: ScalarLike) -> Tensor:
        return Tensor(other - self.data)

    def __isub__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data -= to_arraylike(other)
        return self

    def __mul__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data * to_arraylike(other))

    def __rmul__(self, other: ScalarLike) -> Tensor:
        return Tensor(other * self.data)

    def __imul__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data *= to_arraylike(other)
        return self

    def __truediv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data / to_arraylike(other))

    def __rtruediv__(self, other: ScalarLike) -> Tensor:
        return Tensor(other / self.data)

    def __itruediv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data /= to_arraylike(other)
        return self

    def __floordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data // to_arraylike(other))

    def __rfloordiv__(self, other: ScalarLike) -> Tensor:
        return Tensor(other // self.data)

    def __ifloordiv__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data //= to_arraylike(other)
        return self

    def __pow__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data ** to_arraylike(other))

    def __rpow__(self, other: ScalarLike) -> Tensor:
        return Tensor(other**self.data)

    def __ipow__(self, other: Tensor | ScalarLike) -> Tensor:
        self.data **= to_arraylike(other)
        return self

    def __mod__(self, other: int) -> Tensor:
        return Tensor(self.data % other)

    def __rmod__(self, other: int) -> Tensor:
        return Tensor(other % self.data)

    def __imod__(self, other: int) -> Tensor:
        self.data %= other
        return self

    def __neg__(self) -> Tensor:
        return Tensor(self.data.__neg__())

    def __invert__(self) -> Tensor:
        return Tensor(self.data.__invert__())

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self.data @ other.data)

    def __lt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data < to_arraylike(other))

    def __gt__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data > to_arraylike(other))

    def __le__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data <= to_arraylike(other))

    def __ge__(self, other: Tensor | ScalarLike) -> Tensor:
        return Tensor(self.data >= to_arraylike(other))

    def __eq__(self, other: Any) -> Any:
        return Tensor(self.data == to_arraylike(other))  # type: ignore

    def __ne__(self, other: Any) -> Any:
        return Tensor(self.data != to_arraylike(other))  # type: ignore

    def __len__(self) -> int:
        return self.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __array__(
        self, dtype: Optional[type] = None, copy: Optional[bool] = None
    ) -> numpy.ndarray:
        return self.to_numpy()

    def __bool__(self) -> bool:
        return True

    # ----------------------------------------------------------------------------------
    # DEVICE CONVERSIONS
    # ----------------------------------------------------------------------------------

    def to_device(self, device: Device) -> Tensor:
        """Returns a copy of the tensor on the specified device.

        Parameters
        ----------
        device : Device
            Device to move the tensor to.

        Returns
        -------
        Tensor
            Tensor on the specified device.
        """
        if self.device == device:
            return self

        new_tensor = Tensor(data_to_device(self.data, device))
        if self.grad:
            new_tensor.grad = self.grad.to_device(device)
        return new_tensor

    def ito_device(self, device: Device) -> None:
        """Inplace operation to move the tensor to the specified device.

        Parameters
        ----------
        device : Device
            Device to move the tensor to.
        """
        if self.device == device:
            return

        self.data = data_to_device(self.data, device)
        if self.grad:
            self.grad.ito_device(device)

    def to_cpu(self) -> Tensor:
        """Returns a copy of the tensor on the CPU.

        Returns
        -------
        Tensor
            Tensor on the CPU.
        """
        return self.to_device(cpu)

    def to_cuda(self) -> Tensor:
        """Returns a copy of the tensor on the GPU.

        Returns
        -------
        Tensor
            Tensor on the GPU.
        """
        return self.to_device(cuda)

    # ----------------------------------------------------------------------------------
    # DTYPE CONVERSIONS
    # ----------------------------------------------------------------------------------

    def to_type(self, dtype: DType) -> Tensor:
        """Returns a copy of the tensor with elements cast to the given dtype.

        Parameters
        ----------
        dtype : DType
            Datatype to cast tensor-elements to.

        Returns
        -------
        Tensor
            Tensor with elements cast to the given dtype.
        """
        if self.dtype == dtype:
            return self

        return Tensor(self.data.astype(dtype.t, copy=False))

    def ito_type(self, dtype: DType) -> None:
        """Inplace operation to cast the tensor to the given dtype.

        Parameters
        ----------
        dtype : DType
            Datatype to cast tensor-elements to.
        """
        if self.dtype == dtype:
            return

        self.data = self.data.astype(dtype.t, copy=False)
        if self.grad:
            self.grad.ito_type(dtype)

    def to_int(self) -> Tensor:
        """Returns a copy of the tensor with integer values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.int32` values.
        """
        return self.to_type(int32)

    def to_long(self) -> Tensor:
        """Returns a copy of the tensor with long integer values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.int64` values.
        """
        return self.to_type(int64)

    def to_half(self) -> Tensor:
        """Returns a copy of the tensor with half precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float16` values.
        """
        return self.to_type(float16)

    def to_float(self) -> Tensor:
        """Returns a copy of the tensor with single precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float32` values.
        """
        return self.to_type(float32)

    def to_double(self) -> Tensor:
        """Returns a copy of the tensor with double precision values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.float64` values.
        """
        return self.to_type(float64)

    def to_complex(self) -> Tensor:
        """Returns a copy of the tensor with complex values.

        Returns
        -------
        Tensor
            Tensor with :class:`compyute.complex64` values.
        """
        return self.to_type(complex64)

    # ----------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    def view(self, shape: ShapeLike) -> Tensor:
        """Returns a view of the tensor of a given shape.

        Parameters
        ----------
        shape : ShapeLike
            Shape of the view.

        Returns
        -------
        Tensor
            View of the tensor.
        """
        return Tensor(self.data.reshape(shape))

    def copy(self) -> Tensor:
        """Returns a copy of the tensor.

        Returns
        -------
        Tensor
            Copy of the tensor.
        """
        new_tensor = Tensor(self.data.copy())
        if self.grad:
            new_tensor.grad = self.grad.copy()
        return new_tensor

    def item(self) -> Any:
        """Returns the element as a Python type for 1-element tensors.

        Returns
        -------
        Any
            Tensor element.
        """
        return self.data.item()

    def to_numpy(self) -> numpy.ndarray:
        """Returns the tensor data as a NumPy array.

        Returns
        -------
        numpy.ndarray
            NumPy array of the tensor data.
        """
        return self.to_cpu().data

    def to_list(self) -> list:
        """Returns the tensor data as a list.

        Returns
        -------
        list
            List of the tensor data.
        """
        return self.data.tolist()

    def to_contiguous(self) -> Tensor:
        """Returns a C-contiguous tensor.

        Returns
        -------
        Tensor
            C-contiguous tensor.
        """
        if self.contiguous:
            return self
        return Tensor(self.device.module.ascontiguousarray(self.data))

    # ----------------------------------------------------------------------------------
    # UNARY TENSOR OPS
    # ----------------------------------------------------------------------------------

    def abs(self) -> Tensor:
        """Computes the element-wise absolute value of a tensor.

        See Also
        --------
        :func:`compyute.abs`
        """
        return Tensor(abs(self.data))

    def all(self, dim: Optional[DimLike] = None, *, keepdims: bool = False) -> Tensor:
        """Returns ``True`` if all elements in the tensor are ``True`` over given dimensions.

        See Also
        --------
        :func:`compyute.all`
        """
        return Tensor(self.data.all(dim, keepdims=keepdims))

    def any(self, dim: Optional[DimLike] = None, *, keepdims: bool = False) -> Tensor:
        """Returns ``True`` if at least one element in the tensor is ``True`` over given dimensions.

        See Also
        --------
        :func:`compyute.any`
        """
        return Tensor(self.data.any(dim, keepdims=keepdims))

    def argmax(self, dim: Optional[int] = None, *, keepdims: bool = False) -> Tensor:
        """Returns the indices of maximum values along a given dimension.

        See Also
        --------
        :func:`compyute.argmax`
        """
        return Tensor(self.data.argmax(dim, keepdims=keepdims))

    def imag(self) -> Tensor:
        """Returns the imaginary part of a complex tensor.

        See Also
        --------
        :func:`compyute.imag`
        """
        return Tensor(self.data.imag)

    def max(self, dim: Optional[DimLike] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the maximum of tensor elements over a given dimension.

        See Also
        --------
        :func:`compyute.max`
        """
        return Tensor(self.data.max(dim, keepdims=keepdims))

    def mean(self, dim: Optional[DimLike] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the mean of tensor elements over a given dimension.

        See Also
        --------
        :func:`compyute.mean`
        """
        return Tensor(self.data.mean(dim, keepdims=keepdims))

    def min(self, dim: Optional[DimLike] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the minimum of tensor elements over a given dimension.

        See Also
        --------
        :func:`compyute.min`
        """
        return Tensor(self.data.min(dim, keepdims=keepdims))

    def permute(self, dims: tuple[int, ...]) -> Tensor:
        """Returns a view of the tensor with permuted dimensions.

        See Also
        --------
        :func:`compyute.permute`
        """
        return Tensor(self.data.transpose(dims))

    def real(self) -> Tensor:
        """Returns the real part of a complex tensor.

        See Also
        --------
        :func:`compyute.real`
        """
        return Tensor(self.data.real)

    def squeeze(self) -> Tensor:
        """Returns a tensor with single-dimensional dimensions removed.

        See Also
        --------
        :func:`compyute.squeeze`
        """
        return Tensor(self.data.squeeze())

    def std(self, dim: Optional[DimLike] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the standard deviation of tensor elements over a given dimension.

        See Also
        --------
        :func:`compyute.std`
        """
        return Tensor(self.data.std(dim, keepdims=keepdims))

    def sum(self, dim: Optional[DimLike] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the sum of tensor elements over a given dimension.

        See Also
        --------
        :func:`compyute.sum`
        """
        return Tensor(self.data.sum(dim, keepdims=keepdims))

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        """Returns a view of the tensor with its dimensions transposed.

        See Also
        --------
        :func:`compyute.transpose`
        """
        return Tensor(self.data.swapaxes(dim1, dim2))

    def var(
        self,
        dim: Optional[DimLike] = None,
        *,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> Tensor:
        """Computes the variance of tensor elements over a given dimension.

        See Also
        --------
        :func:`compyute.var`
        """
        return Tensor(self.data.var(dim, ddof=ddof, keepdims=keepdims))


def to_arraylike(value: Any) -> ArrayLike | ScalarLike:
    """Converts a value to an array like."""
    if isinstance(value, Tensor):
        return value.data
    return value
