from typing import Callable

import numpy as np
import torch

activation_factory = {
    "identity": torch.nn.Identity,
    "elu": torch.nn.ELU,
    "hard_shrink": torch.nn.Hardshrink,
    "hard_tanh": torch.nn.Hardtanh,
    "leaky_relu": torch.nn.LeakyReLU,
    "logsigmoid": torch.nn.LogSigmoid,
    "gelu": torch.nn.GELU,
    "prelu": torch.nn.PReLU,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "rrelu": torch.nn.RReLU,
    "selu": torch.nn.SELU,
    "celu": torch.nn.CELU,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
    "soft_shrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "tanh_shrink": torch.nn.Tanhshrink,
    "threshold": torch.nn.Threshold,
    "softmin": torch.nn.Softmin,
    "softmax": torch.nn.Softmax,
    "logsoftmax": torch.nn.LogSoftmax,
    "swish": torch.nn.SiLU,
}


def elu_gradient(act_fn: torch.nn.ELU, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the ELU activation function.

    Args:
        act_fn (torch.nn.ELU): ELU activation.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.where(x > 0, torch.ones_like(x), act_fn.alpha * np.exp(x))


def hard_shrink_gradient(
    act_fn: torch.nn.Hardshrink, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the hard shrink activation function.

    Args:
        act_fn (torch.nn.Hardshrink): hard shrink activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.where(
        torch.logical_and(x > -act_fn.lambd, x < act_fn.lambd),
        torch.zeros_like(x),
        torch.ones_like(x),
    )


def hard_tanh_gradient(
    act_fn: torch.nn.Hardtanh, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the hard tanh activation function.

    Args:
        act_fn (torch.nn.Hardtanh): hard hyperbolic tangent activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """

    return torch.where(
        torch.logical_and(x > act_fn.min_val, x < act_fn.max_val),
        torch.ones_like(x),
        torch.zeros_like(x),
    )


def leaky_relu_gradient(
    act_fn: torch.nn.LeakyReLU, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the leaky relu activation function.

    Args:
        act_fn (torch.nn.LeakyReLU): leaky ReLU activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    o = torch.ones_like(x)
    return torch.where(x > 0, o, o * act_fn.negative_slope)


def logsigmoid_gradient(
    act_fn: torch.nn.LogSigmoid, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the log sigmoid activation function.

    Args:
        act_fn (torch.nn.LogSigmoid): log sigmoid activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.divide(1, torch.exp(x) + 1)


def prelu_gradient(act_fn: torch.nn.PReLU, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the PReLU activation function.

    Args:
        act_fn (torch.nn.PReLU): PreLU activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    o = torch.ones_like(x)
    return torch.where(x > 0, o, o * act_fn.weight)


def relu_gradient(act_fn: torch.nn.ReLU, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the ReLU activation function.

    Args:
        act_fn (torch.nn.ReLU): ReLU activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def relu6_gradient(act_fn: torch.nn.ReLU6, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the ReLU6 activation function.

    Args:
        act_fn (torch.nn.ReLU6): ReLU6 activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.where(
        torch.logical_and(x > 0, x < 6),
        torch.ones_like(x),
        torch.zeros_like(x),
    )


def selu_gradient(act_fn: torch.nn.SELU, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the SELU activation function.

    Args:
        act_fn (torch.nn.SELU): SELU activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    alpha = 1.6732632423543772848170429916717
    lambd = 1.0507009873554804934193349852946
    return lambd * torch.where(x > 0, torch.ones_like(x), alpha * torch.exp(x))


def celu_gradient(act_fn: torch.nn.CELU, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the CELU activation function.

    Args:
        act_fn (torch.nn.CELU): CELU activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.where(x > 0, torch.ones_like(x), np.exp(x / act_fn.alpha))


def sigmoid_gradient(act_fn: torch.nn.Sigmoid, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the sigmoid activation function.

    Args:
        act_fn (torch.nn.Sigmoid): sigmoid activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    sx = torch.sigmoid(x)
    return sx * (1 - sx)


def softplus_gradient(
    act_fn: torch.nn.Softplus, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the softplus activation function.

    Args:
        act_fn (torch.nn.Softplus): softplus activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.exp(x) / (1 + torch.exp(x))


def soft_shrink_gradient(
    act_fn: torch.nn.Softshrink, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the soft shrink activation function.

    Args:
        act_fn (torch.nn.Softshrink): soft shrink activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return hard_shrink_gradient(act_fn, x)


def softsign_gradient(
    act_fn: torch.nn.Softsign, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the softsign activation function.0

    Args:
        act_fn (torch.nn.Softsign): softsign activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return 1 / torch.square(1 + torch.abs(x))


def tanh_gradient(act_fn: torch.nn.Tanh, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the tanh activation function.

    Args:
        act_fn (torch.nn.Tanh): tanh activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return 1 - torch.square(act_fn(x))


def tanh_shrink_gradient(
    act_fn: torch.nn.Tanhshrink, x: torch.Tensor
) -> torch.Tensor:
    """
    Implementation of the gradient of the tanh shrink activation function.

    Args:
        act_fn (torch.nn.Tanhshrink): hyperbolic tangent shrink activation
            function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    return torch.square(act_fn(x))


def swish_gradient(act_fn: torch.nn.GELU, x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gradient of the swish activation function.

    Args:
        act_fn (torch.nn.GELU): swish (GELU) activation function.
        x (torch.Tensor): tensor.

    Returns:
        torch.Tensor: output tensor.
    """
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


activation_gradient_factory = {
    "elu": elu_gradient,
    "hard_shrink": hard_shrink_gradient,
    "hard_tanh": hard_tanh_gradient,
    "leaky_relu": leaky_relu_gradient,
    "logsigmoid": logsigmoid_gradient,
    "prelu": prelu_gradient,
    "relu": relu_gradient,
    "relu6": relu6_gradient,
    "selu": selu_gradient,
    "celu": celu_gradient,
    "sigmoid": sigmoid_gradient,
    "softplus": softplus_gradient,
    "soft_shrink": soft_shrink_gradient,
    "softsign": softsign_gradient,
    "tanh": tanh_gradient,
    "tanh_shrink": tanh_shrink_gradient,
    "swish": swish_gradient,
}


class NormalizedActivation(torch.nn.Module):
    def __init__(
        self,
        act_str: str,
        f: Callable = lambda x: 0.3 * torch.tanh(x),
        momentum: float = 0.9,
        L: int = 0.8,
        U: int = 1.2,
    ):
        super().__init__()
        """Normalized activations from Peiwen et al. [1]. The idea is quite 
        clever - given observations that, while layer outputs may be 
        normalised, gradients are not, the authors normalise the activation
        functions in such a way that approximates normalised gradients. In 
        short, the  following correction is applied to the normalisation 
        function $a$ to the input $X$. The corrected output, $a'$, is given by 
        the following expression:
        
        $a' = (\lambda + f(\alpha)) * (a(X) - \mu)$,

        where $\lambda$ is a normalization factor (see [1] for details), $f$ 
        is a bounded function to adjust $\lambda$ with learnable parameter 
        $\alpha$, and $\mu$ is the (exponential) average of the activation.
        
        Some parameters and their proper hyperparameter optimization are 
        unclear - L and U are used as relative lower and upper bounds, 
        respectively, for the exponential momentum update. Further 
        clarifications are necessary, but everything else is as close as 
        possible to the original paper.

        [1] https://arxiv.org/abs/2208.13315

        Args:
            act_str (str): string corresponding to the activation function. 
                must be one of ['elu', 'hard_shrink', 'hard_tanh', 
                'leaky_relu', 'logsigmoid', 'prelu', 'relu', 'relu6', 'selu',
                'celu', 'sigmoid', 'softplus', 'soft_shrink', 'softsign', 
                'tanh', 'tanh_shrink', 'swish'].
            f (Callable, 0.3 * F.tanh(x)): bounded function to adjust 
                $\lambda$. Defaults to 0.3 * F.tanh(x), as in the original 
                paper.
            momentum (float, 0.9): momentum to update $\mu$, $\rho$ and 
                $\rho'$. Defaults to 0.9.
            L (float, 0.9): relative lower bound for $rho$ and $rho'$ updates.
                Defaults to 0.9.
            U (float, 1.1): relative upper bound for $rho$ and $rho'$ updates.
                Defaults to 1.1.

        Raises:
            NotImplementedError: raises an error when act_str is not in the 
                list above.
        """
        self.act_str = act_str.lower()
        self.f = f
        self.momentum = momentum
        self.L = L
        self.U = U

        if self.act_str not in activation_factory:
            raise NotImplementedError(
                "activation {} not implemented".format(self.act_str)
            )

        if self.act_str not in activation_gradient_factory:
            raise NotImplementedError(
                "gradient for activation {} not implemented".format(
                    self.act_str
                )
            )

        self.act = activation_factory[self.act_str]()
        self.gradient = activation_gradient_factory[self.act_str]
        self.t = 0

        self.initialize_variables()

    def initialize_variables(self):
        self.alpha = torch.nn.Parameter(torch.as_tensor(np.zeros([1])))
        self.mu = torch.zeros([1])
        self.rho = torch.zeros([1])
        self.rho_dash = torch.zeros([1])

    def momentum_update(self, x_0, x_1):
        m = self.momentum
        return x_0 * m + (1 - m) * torch.pow(x_1, self.t - 1)

    def momentum_update_rho(self, x_0, x_1):
        if self.t == 0:
            return x_1
        elif (x_1 < x_0 * self.L) or (x_1 > x_0 * self.U):
            return x_0
        else:
            return self.momentum_update(x_0, x_1)

    def forward(self, X):
        Y = self.act(X)

        rho = torch.divide(torch.square(Y).mean(), X.var())
        rho = rho.reshape([1])
        self.rho = self.momentum_update_rho(self.rho, rho)

        rho_dash = torch.mean(torch.square(self.gradient(self.act, X)))
        self.rho_dash = self.momentum_update_rho(self.rho_dash, rho_dash)
        self.rho_dash.reshape([1])

        self.mu = self.momentum_update(self.mu, Y.mean())

        lam = torch.sqrt((rho + rho_dash) / (2 * rho * rho_dash))
        lam = lam.reshape([1])
        output = (lam + self.f(self.alpha)) * (Y - self.mu)
        self.t += 1
        return output
