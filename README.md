# Compyute Neural Networks

[![CI/CD](https://github.com/dakofler/Compyute/actions/workflows/tests.yml/badge.svg)](https://github.com/dakofler/Compyute/actions/workflows/tests.yml)

Python deep learning library focused on transparency and readability only using `NumPy` and `CuPy` under the hood.
Gradient computation is implemented from scratch to facilitate understanding of the inner workings of neural networks.

## Installation

You can install Compyute directly from GitHub.
```bash
pip install git+https://github.com/dakofler/Compyute.git
```
As of `CuPy` v13, the package does not require a GPU toolkit to be installed, so `Compyute` can be used on CPU-only machines. If you want to make use of GPUs, make sure to install the CUDA Toolkit following the installation guide of `CuPy` (https://docs.cupy.dev/en/stable/install.html).

## Usage

There are example-notebooks included that show how to use the toolbox and its `Tensor` object.

Similar to `PyTorch`, in `Compyute` a `Tensor`-object represents the central block that keeps track of data and gradients. However, unlike `PyTorch`, `Compyute` does not support autograd to compute gradients (yet?). Instead the computation of gradients is defined within a model's layers and functions.

### Tensors

`Tensors` can be created from nested lists of values. Additionally, a `device` where the tensor should be stored on, and a `dtype` of the tensor data can be specified. `Compyute` provides several methods to change these settings.

```python
import compyute as cp

# create a tensor from a list of lists, the data type is inferred automatically
x = cp.tensor([[4, 5, 6], [7, 8, 9]])

# alternatively, define data types
x = cp.tensor([1, 2, 3], dtype=cp.int32)

# change datatypes
y = x.to_type(cp.float32)
y = x.to_float()
y = x.to_int()

# define the device the tensor is stored on
c = cp.tensor([1, 2, 3], device=cp.cuda)

# change devices
c = c.to_device(cp.cpu)
c = c.to_cpu()
```

The `Tensor` object supports most operations also known from `PyTorch` tensors or `NumPy` arrays. Here are some examples:

```python
# basic arithmetic
z = x + y
z = x - y
z = x * y
z = x / y
x += 10

# operations from linear algebra
z = x @ y
z = cp.inner(x, y)
z = cp.outer(x, y)

# aggregate elements of a tensor
z = cp.sum(x, axis=0)
z = cp.mean(x, axis=0)

# functions for initializing tensors
z = cp.ones(shape=(10, 10))
z = cp.zeros(shape=(10, 10))
z = cp.full(shape=(10, 10), value=99)
z = cp.random.normal(shape=(10, 10))
```

### Building models
Models can be built using predefined modules (such as `Linear` or `ReLU`) and containers (such as `Sequential`). `Compyute` provides a variety of modules such as activation, normalization, linear, convolutional and recurrent layers with more to come. For a list of available modules, please check the docs.

```python
from compyute import nn

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

model.to_device(cp.cuda)  # move model to GPU
```

Alternatively, models can also be built entirely from scratch by defining custom classes that inherit from the `Module` class.

With custom models, the user defines what modules to use and how data and gradients flow through the network. Models are generally composed of one or more `Modules`. 

When inheriting from predefined containers, such as `Sequential`, the forward-method is already defined.

```python
from compyute import nn

class MyConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):

        conv = nn.Conv2D(in_channels, out_channels, kernel_size=3)
        relu = nn.ReLU()
        bn = nn.BatchNorm2D(out_channels)
        
        # pass modules to the `Sequential` base class
        super().__init__(conv, relu, bn)
```

If you want to define a custom forward-method, the `Module` base class can be used.

```python
from compyute import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(16, 3)

    # define the forward pass
    @nn.Module.register_forward
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

    # define the backward pass
    @nn.Module.register_backward
    def backward(self, dy):
        dy = self.lin2.backward(dy)
        dy = self.relu.backward(dy)
        dy = self.lin1.backward(dy)
        return dy
```

### Training models

All modules can be trained and updated using common optimizer algorithms, such as `SGD` or `Adam`. Callbacks offer extended functionality such as tracking the loss-values during training. An easy and approchable way is to use a `Trainer` object.

```python
from compyute.nn.trainer import Trainer
from compyute.nn.trainer.callbacks import EarlyStopping, History, Progressbar

trainer = Trainer(
    model=model,
    optimizer="sgd",
    loss_function="cross_entropy",
    metric_function="accuracy",
    callbacks=[EarlyStopping(), History(), Progressbar()]
)

trainer.train(X_train, y_train, epochs=10)
```

Alternatively, you can write your own training loop.

```python
epochs = 100
batch_size = 32

train_dl = nn.utils.Dataloader((X_train, y_train), batch_size)
val_dl = nn.utils.Dataloader((X_val, y_val), batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optim = nn.optimizers.SGD(model.get_parameters())

for epoch in range(epochs):

    # training
    model.training()
    for x, y in train_dl():

        # forward pass
        y_pred = model(x)
        _ = loss_fn(y_pred, y)

        # backward pass
        optim.reset_grads()  # reset all gradients
        model.backward(loss_fn.backward())  # compute new gradients
        optim.step()  # update parameters
    
    # validiation
    model.inference()
    with nn.no_cache_ctx():  # disable caching of contextual data for gradient computation
        val_loss = 0
        for x, y in val_dl():
            y_pred = model(x)
            val_loss += loss_fn(y_pred, y).item()
        val_loss /= len(val_dl)
    
    print(f"epoch {epoch}: {val_loss=:.4f}")
```

Model checkpoints can be saved and loaded later on.

```python
# save the model state
state = {
    "model": model.get_state_dict(),
    "optimizer": optim.get_state_dict(),
}
cp.save(state, "my_model.cp")

# load the states
model = MyModel()
optim = nn.optimizers.SGD(model.get_parameters())
loaded_state = cp.load("my_model.cp")
model.load_state_dict(loaded_state["model"])
optim.load_state_dict(loaded_state["optimizer"])
```

## Examples
see [Examples](https://github.com/dakofler/Compyute/tree/main/examples).


## Author
Daniel Kofler - AI Research Associate ([dkofler@outlook.com](mailto:dkofler@outlook.com))

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Final Notes
I use this project to gain a deeper understanding of the inner workings of neural networks. Therefore, project is a work in progress and possibly will forever be, as I am planning to constantly add new features and optimizations. The code is by far not perfect, as I am still learning with every new feature that is added. If you have any suggestions or find any bugs, please don't hesitate to contact me.

Cheers,<br>
Daniel
