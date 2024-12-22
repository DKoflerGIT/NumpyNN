"""Parameter optimizers."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal, Optional

from ..backend import Device, cpu
from ..tensor_ops.unary_ops import sqrt
from ..tensors import Tensor
from .parameter import Parameter

__all__ = ["SGD", "Adam", "AdamW", "NAdam"]


class Optimizer(ABC):
    """Optimizer base class.

    Parameters
    ----------
    parameters : Iterable[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    """

    def __init__(
        self, parameters: Optional[Iterable[Parameter]] = None, lr: float = 1e-3
    ) -> None:
        self.lr = lr
        self.t = 1
        self._parameters: list[Parameter] = []
        self._state: dict[int, dict[str, Tensor]] = {}

        if parameters is not None:
            self.set_parameters(parameters)

    def set_parameters(self, parameters: Iterable[Parameter]) -> None:
        """Sets the parameters to optimize.

        Parameters
        ----------
        value : Iterable[Parameter]
            Paramters to optimize.
        """
        ptrs: set[int] = set()
        self._parameters = []

        for p in parameters:
            if p.ptr in ptrs:
                continue
            self._parameters.append(p)
            ptrs.add(p.ptr)

        self._state = {i: {} for i in range(len(self._parameters))}

    def get_state_dict(self) -> dict[str, dict[Any, Any]]:
        """Returns a state dict containing the optimizer state and variables.

        Returns
        -------
        dict[str, dict[Any, Any]]
            State dict containing the optimizer state and variables.
        """
        state = {
            i: {k: v.to_cpu() for k, v in d.items()} for i, d in self._state.items()
        }
        variables = {
            k: v for k, v in vars(self).items() if k not in {"_parameters", "_state"}
        }
        return {"state": state, "variables": variables}

    def load_state_dict(
        self,
        state_dict: dict[str, dict[Any, Any]],
        target_device: Optional[Device] = cpu,
    ) -> None:
        """Loads the optimizer state from a state dict.

        .. note::
            This method only loads the optimizer state, not the parameters.
            Parameters must be assigned additionally.

        Parameters
        ----------
        state_dict : dict[str, dict[Any, Any]]
            State dict containing parameters and buffers.
        target_device : Device, optional
            Device to move the optimizer state to. Defaults to ``cpu``.
        """

        # load state
        self._state = {
            i: {k: v.to_device(target_device) for k, v in d.items()}
            for i, d in state_dict["state"].items()
        }

        # load variables
        for k, v in state_dict["variables"].items():
            setattr(self, k, v)

    def reset_grads(self) -> None:
        """Resets parameter gradients to ``None``."""
        for p in self._parameters:
            p.grad = None

    @abstractmethod
    def step(self) -> None:
        """Updates parameters using their gradients."""


class SGD(Optimizer):
    r"""Updates parameters using Stochastic Gradient Descent (with optional momentum).

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{Stochastic Gradient Descent} \\
            & \text{initialize: } v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 1: } \textbf{if } \lambda > 0 \\
            & \text{ 2: } \hspace{5mm} g_t \leftarrow g_t + \lambda p_{t-1} \\
            & \text{ 3: } \textbf{if } \mu > 0 \\
            & \text{ 4: } \hspace{5mm} v_t \leftarrow \mu v_{t-1} + g_t \\
            & \text{ 5: } \hspace{5mm} \textbf{if } nesterov \\
            & \text{ 6: } \hspace{10mm} g_t \leftarrow g_t + \mu v_t \\
            & \text{ 7: } \hspace{5mm} \textbf{else} \\
            & \text{ 8: } \hspace{10mm} g_t \leftarrow v_t \\
            & \text{ 9: } \hspace{5mm} p_t \leftarrow p_{t-1} - \eta g_t \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\mu` ... momentum
        - :math:`v` ... velocity


    Parameters
    ----------
    parameters : Iterable[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    momentum : float, optional
        Momentum factor. Defaults to ``0``.
    nesterov : bool, optional
        Whether to use the neterov momentum algorithm. Defaults to ``False``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``0``.
    """

    def __init__(
        self,
        parameters: Optional[Iterable[Parameter]] = None,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

    def step(self) -> None:
        """Updates parameters using stochastic gradient descent."""
        for i, p in enumerate(self._parameters):
            if p.grad is None:
                continue

            g = p.grad.copy()

            if self.weight_decay > 0.0:
                g += self.weight_decay * p

            if self.momentum > 0.0:
                v_prev = self._state[i].get("v", 0.0)
                v = self.momentum * v_prev + g
                self._state[i]["v"] = v

                if self.nesterov:
                    g += self.momentum * v
                else:
                    g = v

            delta = self.lr * g
            p.data -= delta.data

        self.t += 1


class Adam(Optimizer):
    r"""Updates parameters following the Adam learning algorithm
    as described by `Kingma et al., 2014 <https://arxiv.org/abs/1412.6980>`_.

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{Adam} \\
            & \text{initialize: } m_0 \leftarrow 0,  v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 1: } \textbf{if } \lambda > 0 \\
            & \text{ 2: } \hspace{5mm} g_t \leftarrow g_t + \lambda p_{t-1} \\
            & \text{ 3: } m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            & \text{ 4: } v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
            & \text{ 5: } \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
            & \text{ 6: } \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t} \\
            & \text{ 7: } p_t \leftarrow p_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\beta_1` ... exponential decay rate for the 1st momentum estimate
        - :math:`\beta_2` ... exponential decay rate for the 2nd momentum estimate
        - :math:`\epsilon` ... constant for numerical stability
        - :math:`m` ... first moment estimate (exponential moving average)
        - :math:`v` ... second moment estimate (squared gradient)

    Parameters
    ----------
    parameters : Iterable[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    beta1 : float, optional
        Exponential decay rate for the 1st momentum estimate. Defaults to ``0.9``.
    beta2 : float, optional
        Exponential decay rate for the 2nd momentum estimate. Defaults to ``0.999``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-08``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``0``.
    """

    def __init__(
        self,
        parameters: Optional[Iterable[Parameter]] = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:

        m_div = 1.0 - self.beta1**self.t
        v_div = 1.0 - self.beta2**self.t

        for i, p in enumerate(self._parameters):
            if p.grad is None:
                continue

            if self.weight_decay == 0.0:
                g = p.grad
            else:
                g = p.grad + self.weight_decay * p

            # first moment estimate (exponential moving average)
            m_prev = self._state[i].get("m", 0.0)
            m = self.beta1 * m_prev + (1.0 - self.beta1) * g
            self._state[i]["m"] = m.copy()

            # second moment estimate (squared gradient)
            v_prev = self._state[i].get("v", 0.0)
            v = self.beta2 * v_prev + (1.0 - self.beta2) * g**2
            self._state[i]["v"] = v.copy()

            m /= m_div
            v /= v_div

            delta = self.lr * m / (sqrt(v) + self.eps)
            p.data -= delta.data

        self.t += 1


class AdamW(Optimizer):
    r"""Updates parameters following the AdamW learning algorithm as described by
    `Loshchilov et al., 2017 <https://arxiv.org/pdf/1711.05101>`_.

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{AdamW} \\
            & \text{initialize: } m_0 \leftarrow 0,  v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 2: } p_t \leftarrow (1 - \eta \lambda) p_{t-1} \\
            & \text{ 3: } m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            & \text{ 4: } v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
            & \text{ 5: } \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
            & \text{ 6: } \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t} \\
            & \text{ 7: } p_t \leftarrow p_t - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\beta_1` ... exponential decay rate for the 1st momentum estimate
        - :math:`\beta_2` ... exponential decay rate for the 2nd momentum estimate
        - :math:`\epsilon` ... constant for numerical stability
        - :math:`m` ... first moment estimate (exponential moving average)
        - :math:`v` ... second moment estimate (squared gradient)

    Parameters
    ----------
    parameters : Iterable[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``1e-3``.
    beta1 : float, optional
        Exponential decay rate for the 1st momentum estimate. Defaults to ``0.9``.
    beta2 : float, optional
        Exponential decay rate for the 2nd momentum estimate. Defaults to ``0.999``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-08``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``1e-2``.
    """

    def __init__(
        self,
        parameters: Optional[Iterable[Parameter]] = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:

        m_div = 1.0 - self.beta1**self.t
        v_div = 1.0 - self.beta2**self.t

        for i, p in enumerate(self._parameters):
            if p.grad is None:
                continue

            p.data *= 1.0 - self.lr * self.weight_decay

            # first moment estimate (exponential moving average)
            prev_m = self._state[i].get("m", 0.0)
            m = self.beta1 * prev_m + (1.0 - self.beta1) * p.grad
            self._state[i]["m"] = m.copy()

            # second moment estimate (squared gradient)
            prev_v = self._state[i].get("v", 0.0)
            v = self.beta2 * prev_v + (1.0 - self.beta2) * p.grad**2
            self._state[i]["v"] = v.copy()

            m /= m_div
            v /= v_div

            delta = self.lr * m / (sqrt(v) + self.eps)
            p.data -= delta.data

        self.t += 1


class NAdam(Optimizer):
    r"""Updates parameters following the NAdam learning algorithm as described by
    `Dozat, 2015 <https://cs229.stanford.edu/proj2015/054_report.pdf>`_.

    .. math::
        \begin{aligned}
            & \rule{115mm}{1pt} \\
            & \textbf{NAdam} \\
            & \text{initialize: } m_0 \leftarrow 0,  v_0 \leftarrow 0 \\
            & \rule{115mm}{1pt} \\
            & \text{ 1: } \textbf{if } \lambda > 0 \\
            & \text{ 2: } \hspace{5mm} g_t \leftarrow g_t + \lambda p_{t-1} \\
            & \text{ 3: } \mu_t = \beta_1 \cdot (1 - \frac{1}{2} 0.96^{t \psi}) \\
            & \text{ 4: } \mu_{t+1} = \beta_1 \cdot (1 - \frac{1}{2} 0.96^{(t + 1) \psi}) \\
            & \text{ 5: } m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            & \text{ 6: } v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
            & \text{ 7: } \hat{m_t} \leftarrow \frac{\mu_{t+1} m_t}{1 - \prod_{i = 1}^{t + 1} \mu_i} + \frac{1 - \mu_t}{1 - \prod_{i = 1}^{t} \mu_i} \\
            & \text{ 8: } \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t} \\
            & \text{ 9: } p_t \leftarrow p_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} \\
            & \rule{115mm}{1pt} \\
        \end{aligned}

    where
        - :math:`p` ... parameter value
        - :math:`g` ... parameter gradient value
        - :math:`\eta` ... learning rate
        - :math:`\lambda` ... weight decay
        - :math:`\psi` ... momentum decay
        - :math:`\beta_1` ... exponential decay rate for the 1st momentum estimate
        - :math:`\beta_2` ... exponential decay rate for the 2nd momentum estimate
        - :math:`\epsilon` ... constant for numerical stability
        - :math:`\mu` ... momentum coefficient
        - :math:`m` ... first moment estimate (exponential moving average)
        - :math:`v` ... second moment estimate (squared gradient)
        

    Parameters
    ----------
    parameters : Iterable[Parameter], optional
        Paramters to optimize. Defaults to ``None``.
    lr : float, optional
        Learning rate. Defaults to ``2e-3``.
    beta1 : float, optional
        Exponential decay rate for the 1st momentum estimate. Defaults to ``0.9``.
    beta2 : float, optional
        Exponential decay rate for the 2nd momentum estimate. Defaults to ``0.999``.
    eps : float, optional
        Constant for numerical stability. Defaults to ``1e-08``.
    weight_decay : float, optional
        Weight decay factor. Defaults to ``0``.
    momentum_decay : float, optional
        Momentum decay factor. Defaults to ``4e-3``.
    """

    def __init__(
        self,
        parameters: Optional[Iterable[Parameter]] = None,
        lr: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 4e-3,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self._mu_prod = 1.0

    def step(self) -> None:
        # momentum coefficient
        mu: float = self.beta1 * (1.0 - 0.5 * 0.96 ** (self.t * self.momentum_decay))
        mu_next: float = self.beta1 * (
            1.0 - 0.5 * 0.96 ** ((self.t + 1) * self.momentum_decay)
        )
        self._mu_prod *= mu
        mu_next_prod = self._mu_prod * mu_next

        m_div = 1.0 - mu_next_prod
        g_div = 1.0 - self._mu_prod
        v_div = 1.0 - self.beta2**self.t

        for i, p in enumerate(self._parameters):
            if p.grad is None:
                continue

            if self.weight_decay == 0.0:
                g = p.grad
            else:
                g = p.grad + self.weight_decay * p

            # first moment estimate (exponential moving average)
            m_prev = self._state[i].get("m", 0.0)
            m = self.beta1 * m_prev + (1.0 - self.beta1) * g
            self._state[i]["m"] = m.copy()

            # second moment estimate (squared gradient)
            v_prev = self._state[i].get("v", 0.0)
            v = self.beta2 * v_prev + (1.0 - self.beta2) * g**2
            self._state[i]["v"] = v.copy()

            m = mu_next * m / m_div + (1.0 - mu) * g / g_div
            v /= v_div

            delta = self.lr * m / (sqrt(v) + self.eps)
            p.data -= delta.data

        self.t += 1


OptimizerLike = Optimizer | Literal["sgd", "adam", "adamw", "nadam"]
OPTIMIZERS = {"sgd": SGD, "adam": Adam, "adamw": AdamW, "nadam": NAdam}


def get_optimizer(optimizer: OptimizerLike) -> Optimizer:
    """Returns an instance of an optimizer."""
    if isinstance(optimizer, Optimizer):
        return optimizer
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer}.")
    return OPTIMIZERS[optimizer]()
