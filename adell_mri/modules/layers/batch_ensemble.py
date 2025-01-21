from typing import Callable

import numpy as np
import torch

from .res_blocks import ResidualBlock2d, ResidualBlock3d
from .utils import unsqueeze_to_target


class BatchEnsemble(torch.nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        n: int,
        in_channels: int,
        out_channels: int,
        adn_fn: Callable = torch.nn.Identity,
        op_kwargs: dict = None,
        res_blocks: bool = False,
    ):
        """Batch ensemble layer. Instantiates a linear/convolutional layer
        (depending on spatial_dim) and, given a forward pass, scales the
        channels before and after the application of the linear/convolutional
        layer. Details in [1].

        [1] https://arxiv.org/abs/2002.06715

        Args:
            spatial_dim (int): number of spatial dimensions (has to be 0, 1, 2
                or 3).
            n (int): size of the ensemble.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            adn_fn (Callable, optional): function applied after
                linear/convolutional operations takes the number of channels
                as input. Defaults to torch.nn.Identity.
            op_kwargs (dict, optional): keyword arguments for
                linear/convolutional operation. Defaults to None.
            res_blocks (bool, optional): use residual blocks instead of normal
                convolutions.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adn_fn = adn_fn
        self.op_kwargs = op_kwargs
        self.res_blocks = res_blocks

        self.correct_kwargs()
        self.initialize_layers()

    def correct_kwargs(self):
        if self.op_kwargs is None:
            if self.spatial_dim == 0:
                self.op_kwargs = {}
            else:
                self.op_kwargs = {"kernel_size": 3}

    def initialize_layers(self):
        if self.res_blocks is False:
            if self.spatial_dim == 0:
                self.mod = torch.nn.Linear(
                    self.in_channels, self.out_channels, **self.op_kwargs
                )
            elif self.spatial_dim == 1:
                self.mod = torch.nn.Conv1d(
                    self.in_channels, self.out_channels, **self.op_kwargs
                )
            elif self.spatial_dim == 2:
                self.mod = torch.nn.Conv2d(
                    self.in_channels, self.out_channels, **self.op_kwargs
                )
            elif self.spatial_dim == 3:
                self.mod = torch.nn.Conv3d(
                    self.in_channels, self.out_channels, **self.op_kwargs
                )
        else:
            if self.spatial_dim == 2:
                self.mod = ResidualBlock2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    **self.op_kwargs,
                )
            elif self.spatial_dim == 3:
                self.mod = ResidualBlock3d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    **self.op_kwargs,
                )
        self.all_weights = torch.nn.ParameterDict(
            {
                "pre": torch.nn.Parameter(
                    torch.as_tensor(
                        np.random.normal(1, 0.1, size=[self.n, self.in_channels]),
                        dtype=torch.float32,
                    )
                ),
                "post": torch.nn.Parameter(
                    torch.as_tensor(
                        np.random.normal(1, 0.1, size=[self.n, self.out_channels]),
                        dtype=torch.float32,
                    )
                ),
            }
        )
        self.adn_op = self.adn_fn(self.out_channels)

    def forward(self, X: torch.Tensor, idx: int = None):
        b = X.shape[0]
        if idx is not None:
            pre = torch.unsqueeze(self.all_weights["pre"][idx], 0)
            post = torch.unsqueeze(self.all_weights["post"][idx], 0)
            X = torch.multiply(
                self.mod(X * unsqueeze_to_target(pre, X)),
                unsqueeze_to_target(post, X),
            )
        elif self.training is True:
            idxs = np.random.randint(self.n, size=b)
            pre = torch.stack([self.all_weights["pre"][idx] for idx in idxs])
            post = torch.stack([self.all_weights["post"][idx] for idx in idxs])
            X = unsqueeze_to_target(pre, X) * X
            X = self.mod(X)
            X = unsqueeze_to_target(post, X) * X
        else:
            all_outputs = []
            for idx in range(self.n):
                pre = torch.unsqueeze(self.all_weights["pre"][idx], 0)
                post = torch.unsqueeze(self.all_weights["post"][idx], 0)
                o = torch.multiply(
                    self.mod(X * unsqueeze_to_target(pre, X)),
                    unsqueeze_to_target(post, X),
                )
                all_outputs.append(o)
            X = torch.stack(all_outputs).mean(0)
        return self.adn_op(X)


class BatchEnsembleWrapper(torch.nn.Module):
    def __init__(
        self,
        mod: torch.nn.Module,
        n: int,
        in_channels: int,
        out_channels: int,
        adn_fn: Callable = torch.nn.Identity,
    ):
        """Batch ensemble layer. Wraps a generic module and applies batch
        ensemble accordingly. Details in [1].

        [1] https://arxiv.org/abs/2002.06715

        Args:
            mod (torch.nn.Module): Torch module.
            n (int): size of the ensemble.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            adn_fn (Callable, optional): function applied after
                linear/convolutional operations takes the number of channels
                as input. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.mod = mod
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adn_fn = adn_fn

        self.initialize_layers()

    def initialize_layers(self):
        self.all_weights = torch.nn.ParameterDict(
            {
                "pre": torch.nn.Parameter(
                    torch.as_tensor(
                        np.random.normal(1, 0.1, size=[self.n, self.in_channels]),
                        dtype=torch.float32,
                    )
                ),
                "post": torch.nn.Parameter(
                    torch.as_tensor(
                        np.random.normal(1, 0.1, size=[self.n, self.out_channels]),
                        dtype=torch.float32,
                    )
                ),
            }
        )
        self.adn_op = self.adn_fn(self.out_channels)

    def forward(
        self,
        X: torch.Tensor,
        idx: int = None,
        mod: torch.nn.Module = None,
        *args,
        **kwargs,
    ):
        if mod is None:
            mod = self.mod
        b = X.shape[0]
        if idx is not None:
            if isinstance(idx, int):
                pre = torch.unsqueeze(self.all_weights["pre"][idx], 0)
                post = torch.unsqueeze(self.all_weights["post"][idx], 0)
                X = mod(X * unsqueeze_to_target(pre, X), *args, **kwargs)
                X = torch.multiply(X, unsqueeze_to_target(post, X))
            elif isinstance(idx, (list, tuple)):
                assert len(idx) == X.shape[0], "len(idx) must be == X.shape[0]"
                pre = self.all_weights["pre"][idx]
                post = self.all_weights["post"][idx]
                X = mod(X * unsqueeze_to_target(pre, X), *args, **kwargs)
                X = torch.multiply(X, unsqueeze_to_target(post, X))
            else:
                raise NotImplementedError("idx has to be int, list or tuple")
        elif self.training is True:
            if self.n == 1:
                idxs = [0]
            else:
                idxs = np.random.randint(self.n, size=b)
            pre = torch.stack([self.all_weights["pre"][idx] for idx in idxs])
            post = torch.stack([self.all_weights["post"][idx] for idx in idxs])
            X = unsqueeze_to_target(pre, X) * X
            X = mod(X, *args, **kwargs)
            X = unsqueeze_to_target(post, X) * X
        else:
            with torch.no_grad():
                all_outputs = []
                for idx in range(self.n):
                    pre = unsqueeze_to_target(
                        torch.unsqueeze(self.all_weights["pre"][idx], 0), X
                    )
                    post = torch.unsqueeze(self.all_weights["post"][idx], 0)
                    o = mod(X * pre)
                    o = torch.multiply(o, unsqueeze_to_target(post, o))
                    all_outputs.append(o)
                X = torch.stack(all_outputs).mean(0)
        return self.adn_op(X)
