compyute
===========
.. currentmodule:: compyute


Tensors
-------
.. autosummary::
    :toctree: ../_generated/compyute
    
    tensor
    Tensor


Tensor Operations
-----------------

Creating and Combining Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions to create new tensors or combine existing tensors.

.. autosummary::
    :toctree: ../_generated/compyute
    
    arange
    empty
    empty_like
    full
    full_like
    identity
    linspace
    ones
    ones_like
    zeros
    zeros_like  


Unary Operations
~~~~~~~~~~~~~~~~

Functions that operate on one tensor.

.. autosummary::
    :toctree: ../_generated/compyute

    abs
    clip
    cos
    cosh
    exp
    fft1d
    fft2d
    histogram
    ifft1d
    ifft2d
    is_nan
    log
    log2
    log10
    real
    round
    sech
    sin
    sinh
    sqrt
    tan
    tanh


Multinary Operations
~~~~~~~~~~~~~~~~~~~~

Functions that operate on two or more tensors.

.. autosummary::
    :toctree: ../_generated/compyute

    allclose
    convolve1d_fft
    convolve2d_fft
    dot
    einsum
    inner
    outer
    tensorprod
    tensorsum


Reducing Operations
~~~~~~~~~~~~~~~~~~~

Functions that aggregate tensors.

.. autosummary::
    :toctree: ../_generated/compyute

    all
    any
    mean
    norm
    prod
    std
    tensorprod
    tensorsum
    sum
    var


Shape Operations
~~~~~~~~~~~~~~~~~~~~

Functions that change tensor shapes.

.. autosummary::
    :toctree: ../_generated/compyute
    
    append
    broadcast_to
    concat
    diagonal
    flatten
    flip
    insert_dim
    movedim
    pad
    pad_to_shape
    permute
    pooling1d
    pooling2d
    repeat1d
    repeat2d
    reshape
    split
    squeeze
    stack
    tile
    transpose


Selecting Operations
~~~~~~~~~~~~~~~~~~~~

Functions that return a subset of a tensor.

.. autosummary::
    :toctree: ../_generated/compyute

    argmax
    get_diagonal
    max
    maximum
    mean
    min
    minimum
    get_diagonal
    tril
    triu
    unique


Data Types
----------

Available data types are:

.. autosummary::
    :toctree: ../_generated/compyute
    
    bool_
    int8
    int16
    int32
    int64
    float16
    float32
    float64
    complex64
    complex128


Utilities:

.. autosummary::
    :toctree: ../_generated/compyute
    
    set_default_dtype
    use_dtype
    

Devices
-------

Available devices are:

.. autosummary::
    :toctree: ../_generated/compyute
    
    cpu
    cuda


Utilities:

.. autosummary::
    :toctree: ../_generated/compyute
    
    Device
    set_default_device
    use_device


Utility Functions
-----------------
.. autosummary::
    :toctree: ../_generated/compyute

    save
    load
    set_debug_mode
    debug


Constants
-----------------
.. data:: newaxis

    A convenient alias for None, useful for indexing arrays.