from copy import deepcopy
from itertools import product
from typing import Sequence

import einops
import numpy as np
import torch

from adell_mri.custom_types import (
    Callable,
    Dict,
    List,
    Size2dOr3d,
    TensorList,
    Tuple,
    Union,
)
from adell_mri.modules.layers.adn_fn import get_adn_fn
from adell_mri.modules.layers.linear_blocks import MLP, MultiHeadSelfAttention
from adell_mri.modules.layers.regularization import ChannelDropout


def move_axis(X: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
    axes = [i for i in range(len(X.shape))]
    if axis1 < 0:
        axis1 = axes[axis1]
    if axis2 < 0:
        axis2 = axes[axis2]
    axes.insert(axis2, axes.pop(axis1))
    return X.permute(tuple(axes))


def einops_rescale(X: torch.Tensor, scale: int = 1) -> torch.Tensor:
    sh = X.shape
    b, c, inner = sh[0], sh[1], sh[2:]
    einops_dict = {"b": b, "c": c}
    size_names = ["h", "w", "d"]
    for i, (coord, s) in enumerate(zip(inner, scale)):
        einops_dict[size_names[i]] = coord // s
        einops_dict[f"p{i+1}"] = s
    if len(sh) == 4:
        einops_str = "b c (h p1) (w p2) -> b (c p1 p2) h w"
    elif len(sh) == 5:
        einops_str = "b c (h p1) (w p2) (d p3) -> b (c p1 p2 p3) h w d"
    return einops.rearrange(X, einops_str, **einops_dict)


def cyclic_shift_batch(X: torch.Tensor, shift: List[int]):
    """
    Applies what the authors from SWIN call a "cyclic shift".

    Args:
        X (:torch.Tensors): input tensor.
        shift (List[int]): size of shift along all dimensions.

    Returns:
        torch.Tensor: rolled input.
    """
    dims = [i for i in range(2, len(shift) + 2)]
    return torch.roll(X, shifts=shift, dims=dims)


def downsample_ein_op_dict(
    ein_op_dict: Dict[str, int], scale: int = 1
) -> Dict[str, int]:
    """
    Downsamples a einops constant dictionary given a downsampling scale
    factor. Useful for UNETR/SWIN applications. This can then be used to
    parametrize the LinearEmbedding.rearrange_inverse operation in a way that
    a new tensor with downsampled resolution and more feature maps is produced.

    Args:
        ein_op_dict (Dict[str,int]): a dictionary containing x,y,z,h,w,d,c
            keys (z and d are optional), as produced by
            LinearEmbedding.get_einop_params.
        scale (int list[int], optional): downsampling scale. Defaults to 1.

    Returns:
        Dict[str,int]: dictionary where the x,y(,z) values are rescaled.
    """
    ein_op_dict = deepcopy(ein_op_dict)
    pairs = (("h", "x"), ("w", "y"), ("d", "z"))
    p = 0
    if isinstance(scale, int):
        scale = [scale] * len(pairs)
    for n_patches, size in pairs:
        if size in ein_op_dict:
            s = scale[p]
            p += 1
            ein_op_dict[size] = ein_op_dict[size] // s
    ein_op_dict["c"] = ein_op_dict["c"] * np.prod(scale)
    return ein_op_dict


def window_partition_custom(
    x: torch.Tensor, window_size: Size2dOr3d
) -> torch.Tensor:
    """
    Reshapes an image/volume batch tensor into smaller image/volumes of
    window_size. Generalizes the implementation in [1] to both images and
    volumes.

    [1] https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

    Args:
        x (torch.Tensor): tensor with shape (b,h,w,(d),c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b,
                  window_size[0],
                  window_size[1],
                  (window_size[2]),c)
    """
    sh = x.shape
    B, in_shape, C = sh[0], sh[1:-1], sh[-1]
    view_sh = [B]
    for s, w in zip(in_shape, window_size):
        view_sh.append(s // w)
        view_sh.append(w)
    view_sh.append(C)
    permute_dims = [
        0,
        *[1 + i * 2 for i in range(len(in_shape))],
        *[2 + i * 2 for i in range(len(in_shape))],
    ]
    permute_dims.append(len(permute_dims))
    x = x.view(*view_sh)
    windows = x.permute(*permute_dims).contiguous().view(-1, *window_size, C)
    return windows


def image_mask_to_attention_mask(
    x: torch.Tensor, window_size: Size2dOr3d
) -> torch.Tensor:
    """
    Reshapes an image/volume batch tensor into smaller image/volumes of
    window_size. Generalizes the implementation in [1] to both images and
    volumes.

    [1] https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

    Args:
        x (torch.Tensor): tensor with shape (b,h,w,(d),c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b,
                  window_size[0],
                  window_size[1],
                  (window_size[2]),c)
    """
    x = x.squeeze(0).squeeze(-1)
    in_shape = x.shape
    einops_dict = {}
    sizes = ["h", "w", "d"]
    for i in range(len(window_size)):
        einops_dict[f"w{i + 1}"] = window_size[i]
        einops_dict[sizes[i]] = in_shape[i] // window_size[i]
    if len(in_shape) == 2:
        einops_expression = "(w1 h) (w2 w) -> (h w) (w1 w2)"
    if len(in_shape) == 3:
        einops_expression = "(w1 h) (w2 w) (w3 d) -> (h w d) (w1 w2 w3)"
    mask_windows = einops.rearrange(x, einops_expression, **einops_dict)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attention_mask


def generate_mask(
    image_size: Size2dOr3d, window_size: Size2dOr3d, shift_size: Size2dOr3d
) -> torch.Tensor:
    """
Masks the attention in a self-attention module in shifted inputs. A
    generalization of the mask generation in [1] for 2 and 3 dimensions.

    [1] https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

    Args:
        image_size (Size2dOr3d): size of image.
        window_size (Size2dOr3d): size of window.
        shift_size (Size2dOr3d): size of the shift.

    Returns:
        torch.Tensor: tensor used to mask attention outputs.
    """
    if isinstance(window_size, list) is False:
        window_size = [window_size for _ in image_size]
    if isinstance(shift_size, list) is False:
        shift_size = [shift_size for _ in image_size]
    attn_mask = None
    if any([x > 0 for x in shift_size]):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, *image_size, 1))  # 1,*image_size,1
        slices = [
            (slice(0, -w), slice(-w, -s), slice(-s, None))
            for w, s in zip(window_size, shift_size)
        ]
        cnt = 0
        for x in product(*slices):
            if len(x) == 2:
                img_mask[:, x[0], x[1], :] = cnt
            if len(x) == 3:
                img_mask[:, x[0], x[1], x[2], :] = cnt
            cnt += 1
        attn_mask = image_mask_to_attention_mask(img_mask, window_size)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def sinusoidal_positional_encoding(n_tokens, dim_size):
    token_range = np.arange(0, n_tokens)[:, np.newaxis]
    dim_range = np.arange(0, dim_size)[np.newaxis, :]

    radians = token_range / (10000 ** (2 * dim_range / dim_size))
    output = np.zeros((n_tokens, dim_size))
    output[:, ::2] = np.sin(radians)[:, ::2]
    output[:, 1::2] = np.cos(radians)[:, 1::2]
    return output


class SliceLinearEmbedding(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        image_size: Tuple[int, int, int],
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
        embedding_size: int = None,
        out_dim: int = None,
        dropout_rate: float = 0.0,
        embed_method: str = "linear",
        use_class_token: bool = False,
        n_registers: int = 0,
        learnable_embedding: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.use_class_token = use_class_token
        self.n_registers = n_registers
        self.learnable_embedding = learnable_embedding

        self.drop_op = torch.nn.Dropout(self.dropout_rate)

        self.n_patches = int(
            np.prod(
                [
                    s // p
                    for s, p in zip(self.image_size[:2], self.patch_size[:2])
                ]
            )
        )
        self.embedding_size = int(np.prod([*patch_size[:2], in_channels]))

        self.init_linear_layers_if_necessary()
        self.init_conv_layers_if_necessary()
        self.init_positional_embedding()
        self.init_class_token_if_necessary()
        self.init_registers_if_necessary()

    def init_conv_layers_if_necessary(self):
        if self.embed_method == "convolutional":
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,
                    self.true_n_features,
                    self.patch_size[:2],
                    stride=self.patch_size[:2],
                )
            )
        else:
            self.conv = None

    def init_class_token_if_necessary(self):
        if self.use_class_token is True:
            self.class_token = torch.nn.Parameter(
                torch.zeros([1, 1, 1, self.true_n_features])
            )

    def init_registers_if_necessary(self):
        if self.n_registers > 0:
            self.registers = torch.nn.Parameter(
                torch.zeros([1, self.n_registers, self.true_n_features])
            )

    def linearize_image_slices(self, image: torch.Tensor) -> torch.Tensor:
        h, w, s = self.image_size
        c = self.in_channels
        x, y, z = self.patch_size
        if self.embed_method == "linear":
            return einops.rearrange(
                image,
                "b c (h x) (w y) s -> b s (h w) (x y c)",
                x=x,
                h=h // x,
                y=y,
                w=w // y,
                c=c,
                s=s,
            )
        else:
            image = einops.rearrange(image, "b c h w s -> (b s) c h w")
            image = self.conv(image)
            return einops.rearrange(
                image,
                "(b s) c h w -> b s (h w) c",
                h=h // x,
                w=w // y,
                c=self.true_n_features,
                s=s,
            )

    def init_positional_embedding(self):
        if self.learnable_embedding is True:
            self.positional_embedding = torch.nn.Parameter(
                torch.zeros(
                    [
                        1,
                        self.image_size[2],
                        self.n_patches,
                        self.true_n_features,
                    ]
                )
            )
            torch.nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        else:
            sin_embed = sinusoidal_positional_encoding(
                self.n_patches, self.true_n_features
            ).reshape(1, 1, self.n_patches, self.true_n_features)
            self.positional_embedding = torch.nn.Parameter(
                torch.as_tensor(sin_embed, dtype=torch.float32),
                requires_grad=False,
            )

    def init_linear_layers_if_necessary(self):
        """
Initialises a linear layers to convert to and from out_dim. This
        allows for the linear embedding to have a different output without
        affecting the size of everything else in the image before and after
        the linear embedding.
        """
        self.map_to_out = torch.nn.Identity()
        self.map_to_in = torch.nn.Identity()
        if self.out_dim is not None and self.out_dim != self.embedding_size:
            if self.embed_method == "linear":
                self.map_to_out = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.embedding_size),
                    torch.nn.Linear(self.embedding_size, self.out_dim),
                )
            self.map_to_in = torch.nn.Linear(self.out_dim, self.embedding_size)
            self.true_n_features = self.out_dim
        else:
            self.true_n_features = self.embedding_size

    def forward(self, X):
        """
Forward pass.

        Args:
            X (torch.Tensor): a tensor with shape
                [-1,self.in_channels,*self.image_size]

        Returns:
            torch.Tensor: the embedding of X, with size
                [X.shape[0],self.n_patches,self.true_n_features].
        """
        # output should always be [X.shape[0],self.n_patches,self.true_n_features]
        b, s = X.shape[0], X.shape[-1]
        X = self.linearize_image_slices(X)
        X = self.map_to_out(X)
        X = X + self.positional_embedding
        if self.use_class_token is True:
            class_token = einops.repeat(
                self.class_token, "() () n e -> b s n e", b=b, s=s
            )
            X = torch.concat([class_token, X], 2)
        if self.n_registers > 0:
            registers = einops.repeat(
                self.registers, "() n e -> b s n e", b=b, s=s
            )
            X = torch.concat([registers, X], 2)
        return self.drop_op(X)


class LinearEmbedding(torch.nn.Module):
    """
Linearly embeds images as described in the vision transformer paper
    [1]. Essentially, it rearranges a given image of size [b,c,h,w] with a
    given patch size [p1,p2] such that the output is
    [b,c*(h//p1)*(w//p2),p1*p2]. This class also features the operations
    necessary to i) reverse this operation and ii) "reverse" this operation
    considering that the output has to be a downscaled version of the input
    as described in the UNETR paper [2]. The convolutional embeding method
    was inspired by the PatchEmbeddingBlock in MONAI [3] but I reworked it
    so that it *always* uses einops, making operations more easily
    reversible.

    Here, I also included an argument that enables the support of windows as
    in SWIN.

    [1] https://arxiv.org/pdf/2010.11929.pdf
    [2] https://arxiv.org/pdf/2103.10504.pdf
    [3] https://docs.monai.io/en/stable/_modules/monai/networks/blocks/patchembedding.html
    [4] https://arxiv.org/pdf/2103.14030.pdf
    """

    def __init__(
        self,
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        in_channels: int,
        out_dim: int = None,
        window_size: Size2dOr3d = None,
        dropout_rate: float = 0.0,
        embed_method: str = "linear",
        use_pos_embed: bool = True,
        use_class_token: bool = False,
        n_registers: int = 0,
        learnable_embedding: bool = True,
        channel_to_token: bool = False,
        channels_last: bool = False,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            in_channels (int): number of channels in the input image.
            out_dim (int): number of dimensions in output.
            window_size (Size2dOr3d, optional): window size for windowed
                multi-head attention. Defaults to None (no windowing)
            dropout_rate (float, optional): dropout rate of the dropout
                operation that is applied after the sum of the linear
                embeddings with the positional embeddings. Defaults to 0.0.
            embed_method (str, optional): embedding method. Defaults to
                "linear".
            use_pos_embed (bool, optional): includes positional embedding.
                Defaults to True.
            use_class_token (bool, optional): whether a class token should be
                used. Defaults to False.
            n_registers (int, optional): number of registers to use as
                suggested in Darcet (2024). Defaults to 0.
            learnable_embedding (bool, optional): if embedding is
                non-trainable, a sinusoidal positional embedding is used.
                Defaults to True.
            channel_to_token (bool, optional): embeds the channel dimension as
                tokens rather than as part of the embedding. Works only with
                "linear" embedding. Defaults to False.
            channels_last (bool, optional): whether the channels in input and
                output tensors are the last dimension. Defaults to False.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.use_pos_embed = use_pos_embed
        self.use_class_token = use_class_token
        self.n_registers = n_registers
        self.learnable_embedding = learnable_embedding
        self.channel_to_token = channel_to_token
        self.channels_last = channels_last

        embed_methods = ["linear", "convolutional"]
        assert (
            self.embed_method in embed_methods
        ), "embed_method must be one of {}".format(embed_methods)
        assert len(self.image_size) == len(
            self.patch_size
        ), "image_size and patch_size should have the same length"
        assert len(self.image_size) in [
            2,
            3,
        ], "len(image_size) has to be 2 or 3"
        assert all(
            [x % y == 0 for x, y in zip(self.image_size, self.patch_size)]
        ), "the elements of image_size should be divisible by those in patch_size"
        if self.channel_to_token is True:
            assert (
                self.embed_method == "linear"
            ), "embed_method must be 'linear' if channel_to_token == True"

        self.calculate_parameters()
        self.init_conv_if_necessary()
        self.init_linear_layers_if_necessary()
        self.init_dropout()
        self.initialize_classification_token()
        self.initialize_registers()
        self.initialize_positional_embeddings()
        self.get_einop_params()
        self.linearized_dim = [-1, self.n_patches, self.n_features]

    @property
    def true_n_features(self):
        return self.out_dim if self.out_dim else self.n_features

    def init_conv_if_necessary(self):
        """
Initializes a convolutional if embed_method == "convolutional" """
        if self.embed_method == "convolutional":
            if self.n_dims == 2:
                self.conv = torch.nn.Conv2d(
                    self.in_channels,
                    self.true_n_features,
                    self.patch_size,
                    stride=self.patch_size,
                )
            elif self.n_dims == 3:
                self.conv = torch.nn.Conv3d(
                    self.in_channels,
                    self.true_n_features,
                    self.patch_size,
                    stride=self.patch_size,
                )

    def init_linear_layers_if_necessary(self):
        """
Initialises a linear layers to convert to and from out_dim. This
        allows for the linear embedding to have a different output without
        affecting the size of everything else in the image before and after
        the linear embedding.
        """
        self.map_to_out = torch.nn.Identity()
        self.map_to_in = torch.nn.Identity()
        if self.out_dim is not None and self.out_dim != self.n_features:
            if self.embed_method == "linear":
                self.map_to_out = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.n_features),
                    torch.nn.Linear(self.n_features, self.out_dim),
                )
            self.map_to_in = torch.nn.Linear(self.out_dim, self.n_features)

    def init_dropout(self):
        """
Initialises the dropout operation that is applied after adding the
        positional embeddings to the sequence embeddings.
        """
        self.drop_op = torch.nn.Dropout(self.dropout_rate)

    def calculate_parameters(self):
        """
Calculates a few handy parameters for the linear embedding."""
        self.n_dims = len(self.image_size)
        if self.window_size is None:
            self.n_patches_split = [
                x // y for x, y in zip(self.image_size, self.patch_size)
            ]
        else:
            # number of patches will be smaller but the number of features
            # remains the same
            self.n_windows = [
                x // y for x, y in zip(self.image_size, self.window_size)
            ]
            self.n_patches_split = [
                x // z // y
                for x, y, z in zip(
                    self.image_size, self.patch_size, self.n_windows
                )
            ]
        if self.channel_to_token is True:
            extra_patches = self.in_channels
            extra_features = 1
        else:
            extra_patches = 1
            extra_features = self.in_channels
        self.n_patches = int(np.prod(self.n_patches_split) * extra_patches)
        self.n_features = int(np.prod(self.patch_size) * extra_features)

    def initialize_classification_token(self):
        """
Initializes the classification token."""
        if self.use_class_token is True:
            self.class_token = torch.nn.Parameter(
                torch.zeros([1, 1, self.true_n_features])
            )

    def initialize_registers(self):
        """
Initializes the transformer registers."""
        if self.n_registers > 0:
            self.registers = torch.nn.Parameter(
                torch.zeros([1, self.n_registers, self.true_n_features])
            )

    def initialize_positional_embeddings(self):
        """
Initilizes the positional embedding with a truncated normal
        distribution.
        """
        if self.use_pos_embed is True:
            if self.learnable_embedding is True:
                self.positional_embedding = torch.nn.Parameter(
                    torch.rand(1, self.n_patches, self.true_n_features)
                )
                torch.nn.init.trunc_normal_(self.positional_embedding, std=0.02)
            else:
                sin_embed = sinusoidal_positional_encoding(
                    self.n_patches, self.true_n_features
                )[np.newaxis, :, :]
                self.positional_embedding = torch.nn.Parameter(
                    torch.as_tensor(sin_embed, dtype=torch.float32),
                    requires_grad=False,
                )

    def einops_tuple(self, einops_str):
        if isinstance(einops_str, list):
            if len(einops_str) > 1:
                return "({})".format(" ".join(einops_str))
            else:
                return einops_str[0]
        else:
            return einops_str

    def get_linear_einop_params(self):
        lh = ["b", "c", ["h", "x"], ["w", "y"]]
        if self.channel_to_token is True:
            rh = ["b", ["h", "w", "c"], ["x", "y"]]
        else:
            rh = ["b", ["h", "w"], ["x", "y", "c"]]
        einop_dict = {
            k: int(s) for s, k in zip(self.patch_size, ["x", "y", "z"])
        }
        einop_dict.update(
            {k: int(s) for s, k in zip(self.n_patches_split, ["h", "w", "d"])}
        )
        einop_dict["c"] = self.in_channels
        if self.n_dims == 3:
            lh.append(["d", "z"])
            rh[-2].append("d")
            rh[-1].insert(2, "z")
        if self.window_size is not None:
            windowing_vars = ["w{}".format(i + 1) for i in range(self.n_dims)]
            for i, w in enumerate(windowing_vars):
                lh[i + 2].insert(0, w)
            rh.insert(1, self.einops_tuple(windowing_vars))
            einop_dict.update(
                {k: int(w) for w, k in zip(self.n_windows, ["w1", "w2", "w3"])}
            )
        return lh, rh, einop_dict

    def get_conv_einop_params(self):
        lh = ["b", "c", ["h"], ["w"]]
        rh = ["b", ["h", "w"], "c"]
        einop_dict = {}
        einop_dict["c"] = self.true_n_features
        einop_dict.update(
            {k: int(s) for s, k in zip(self.n_patches_split, ["h", "w", "d"])}
        )
        if self.n_dims == 3:
            lh.append(["d"])
            rh[-2].append("d")
        if self.window_size is not None:
            windowing_vars = ["w{}".format(i + 1) for i in range(self.n_dims)]
            for i, w in enumerate(windowing_vars):
                lh[i + 2].insert(0, w)
            rh.insert(1, self.einops_tuple(windowing_vars))
            einop_dict.update(
                {k: int(w) for w, k in zip(self.n_windows, ["w1", "w2", "w3"])}
            )
        return lh, rh, einop_dict

    def get_einop_params(self):
        """
Defines all necessary einops constants. This reduces the amount of
        inference that einops.rearrange has to do internally and ensurest that
        this operation is a bit easier to inspect.
        """
        (
            self.lh_l,
            self.rh_l,
            self.einop_dict_l,
        ) = self.get_linear_einop_params()
        self.lh_c, self.rh_c, self.einop_dict_c = self.get_conv_einop_params()

        if self.channels_last is True:
            self.lh_l.remove("c")
            self.lh_c.remove("c")
            self.lh_l.append("c")
            self.lh_c.append("c")

        self.einop_str_l = "{lh} -> {rh}".format(
            lh=" ".join([self.einops_tuple(x) for x in self.lh_l]),
            rh=" ".join([self.einops_tuple(x) for x in self.rh_l]),
        )
        self.einop_inv_str_l = "{lh} -> {rh}".format(
            lh=" ".join([self.einops_tuple(x) for x in self.rh_l]),
            rh=" ".join([self.einops_tuple(x) for x in self.lh_l]),
        )

        self.einop_str_c = "{lh} -> {rh}".format(
            lh=" ".join([self.einops_tuple(x) for x in self.lh_c]),
            rh=" ".join([self.einops_tuple(x) for x in self.rh_c]),
        )
        self.einop_inv_str_c = "{lh} -> {rh}".format(
            lh=" ".join([self.einops_tuple(x) for x in self.rh_c]),
            rh=" ".join([self.einops_tuple(x) for x in self.lh_c]),
        )

        if self.embed_method == "linear":
            self.lh = self.lh_l
            self.rh = self.rh_l
            self.einop_dict = self.einop_dict_l
            self.einop_str = self.einop_str_l
            self.einop_inv_str = self.einop_inv_str_l
        elif self.embed_method == "convolutional":
            self.lh = self.lh_c
            self.rh = self.rh_c
            self.einop_dict = self.einop_dict_c
            self.einop_str = self.einop_str_c
            self.einop_inv_str = self.einop_inv_str_c

        self.einop_rearrange = einops.layers.torch.Rearrange(
            self.einop_str, **self.einop_dict
        )
        self.einop_rearrange_inv = einops.layers.torch.Rearrange(
            self.einop_inv_str, **self.einop_dict
        )

    def get_downsample_str_and_params(
        self, scale: list[int]
    ) -> tuple[str, dict[str, int]]:
        lh_l = deepcopy(self.lh_l)
        rh_l = deepcopy(self.rh_l)
        einop_dict_l = deepcopy(self.einop_dict_l)
        lh_l[1] = [lh_l[1]]
        size_list = ["x", "y", "z"]
        for i in range(len(self.image_size)):
            k = f"ds{i+1}"
            lh_l[1].append(k)
            if self.window_size is not None:
                rh_l[3].insert(2 * i + 1, k)
            else:
                rh_l[2].insert(2 * i + 1, k)
            einop_dict_l[k] = scale[i]
            einop_dict_l[size_list[i]] = einop_dict_l[size_list[i]] // scale[i]
        if self.channels_last is True:
            lh_l.remove("c")
            lh_l.append("c")
        einops_inv_str = "{rh} -> {lh}".format(
            lh=" ".join([self.einops_tuple(x) for x in lh_l]),
            rh=" ".join([self.einops_tuple(x) for x in rh_l]),
        )
        return einop_dict_l, einops_inv_str

    def rearrange(self, X: torch.Tensor) -> torch.Tensor:
        """
Applies einops.rearrange given the parameters inferred in
        self.get_einop_params.

        Args:
            X (torch.Tensor): a tensor of size (b,c,h,w,(d))

        Returns:
            torch.Tensor: a tensor of size (b,h*x,w*y,(d*z))
        """
        if self.embed_method == "linear":
            X = self.einop_rearrange(X)
            X = self.map_to_out(X)
        elif self.embed_method == "convolutional":
            X = self.einop_rearrange(X)
        return X

    def rearrange_inverse_basic(self, X: torch.Tensor) -> torch.Tensor:
        """
Reverses the self.rearrange operation using the parameters inferred
        in self.get_einop_params.

        Args:
            X (torch.Tensor): a tensor of size (b,h*x,w*y,(d*z))
            kwargs: arguments that will be appended to self.einop_dict (only
                works with embed_method == "linear").

        Returns:
            x torch.Tensor: a tensor of size (b,c,h,w,(d))
        """
        if self.embed_method == "linear":
            X = self.map_to_in(X)
            X = self.einop_rearrange_inv(X)
        elif self.embed_method == "convolutional":
            X = self.einop_rearrange_inv(X)
        return X

    def rearrange_inverse(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
Reverses the self.rearrange operation using the parameters inferred
        in self.get_einop_params.

        Args:
            X (torch.Tensor): a tensor of size (b,h*x,w*y,(d*z))
            kwargs: arguments that will be appended to self.einop_dict (only
                works with embed_method == "linear").

        Returns:
            x torch.Tensor: a tensor of size (b,c,h,w,(d))
        """
        X = self.map_to_in(X)
        einop_dict = deepcopy(self.einop_dict_l)
        for k in kwargs:
            einop_dict[k] = kwargs[k]
        if self.embed_method == "linear":
            X = einops.rearrange(X, self.einop_inv_str, **einop_dict)
        elif self.embed_method == "convolutional":
            X = einops.rearrange(X, self.einop_inv_str_l, **einop_dict)
        return X

    def rearrange_rescale(
        self, X: torch.Tensor, scale: int | list[int]
    ) -> torch.Tensor:
        """
        Reverses the self.rearrange operation using the parameters inferred
        in self.get_einop_params but rescales the resolution so that the "extra"
        features are stacked on the channel dimension (for UNETR, etc.).

        Args:
            X (torch.Tensor): a tensor of size (b,h*x,w*y,(d*z))
            scale (int, list[int]): factor by which resolution will be
                downsampled. If int, all dimensions are scaled equally,
                if list, each dimension is scaled separately.

        Returns:
            x torch.Tensor: a tensor of size (b,c,h,w,(d))
        """
        X = self.map_to_in(X)
        if isinstance(scale, int):
            scale = [scale] * len(self.image_size)
        if self.embed_method == "linear":
            einop_dict, einop_str = self.get_downsample_str_and_params(scale)
            X = einops.rearrange(X, einop_str, **einop_dict)
        elif self.embed_method == "convolutional":
            image_size = [
                self.image_size[i] // scale[i]
                for i in range(len(self.image_size))
            ]
            in_channels = self.in_channels * np.prod(scale)
            X = X.reshape(-1, in_channels, *image_size)
        return X

    def forward(
        self, X: torch.Tensor, no_pos_embed: bool = False
    ) -> torch.Tensor:
        """
Forward pass.

        Args:
            X (torch.Tensor): a tensor with shape
                [-1,self.in_channels,*self.image_size]
            no_pos_embed (bool, optional): skips the addition of the positional
                embedding.

        Returns:
            torch.Tensor: the embedding of X, with size
                [X.shape[0],self.n_patches,self.true_n_features].
        """
        # output should always be [X.shape[0],self.n_patches,self.true_n_features]
        if self.embed_method == "convolutional":
            if self.channels_last is True:
                X = move_axis(X, -1, 1)
                X = self.conv(X)
                X = move_axis(X, 1, -1)
            else:
                X = self.conv(X)
        X = self.rearrange(X)
        if (no_pos_embed is False) and (self.use_pos_embed is True):
            X = X + self.positional_embedding
        if self.use_class_token is True:
            class_token = einops.repeat(
                self.class_token, "() n e -> b n e", b=X.shape[0]
            )
            X = torch.concat([class_token, X], 1)
        if self.n_registers > 0:
            registers = einops.repeat(
                self.registers, "() n e -> b n e", b=X.shape[0]
            )
            X = torch.concat([registers, X], 1)
        return self.drop_op(X)


class TransformerBlock(torch.nn.Module):
    """
    Transformer block. Built on top of a multi-head attention (MHA), it
    can be summarised as:

                                |---------------|
    input -> MHA -> Add -> LayerNorm -> MLP -> Add -> LayerNorm -> Output
        |------------|

    First introduced, to the best of my knowledge, in [1].

    [1] https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        input_dim_primary: int,
        attention_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        mlp_structure: List[int] = [128, 128],
        dropout_rate: float = 0.0,
        window_size: Size2dOr3d = None,
        adn_fn: Callable = get_adn_fn(1, "identity", "gelu"),
    ):
        """
        Args:
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (List[int], optional): hidden layer structure.
                Should be a list of ints. Defaults to [32,32].
            dropout_rate (float, optional): dropout rate, applied to the output
                of each sub-layer. Defaults to 0.0.
            window_size (bool, optional): window_size for windowed W-MSA.
                Defaults to None (regular block of transformers).
            adn_fn (Callable, optional): function that returns a
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to
                get_adn_fn(1,"identity","gelu").
        """
        super().__init__()
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_structure = mlp_structure
        self.dropout_rate = dropout_rate
        self.window_size = window_size
        self.adn_fn = adn_fn

        self.mha = MultiHeadSelfAttention(
            self.input_dim_primary,
            self.attention_dim,
            self.hidden_dim,
            self.input_dim_primary,
            window_size=self.window_size,
            dropout_rate=self.dropout_rate,
            n_heads=self.n_heads,
        )

        self.init_drop_ops()
        self.init_layer_norm_ops()
        self.init_mlp()

    def init_drop_ops(self):
        """
Initializes the dropout operations."""
        self.drop_op_1 = torch.nn.Dropout(self.dropout_rate)
        self.drop_op_2 = torch.nn.Dropout(self.dropout_rate)

    def init_layer_norm_ops(self):
        """
Initializes the MLP in the last step of the transformer."""
        self.norm_op_1 = torch.nn.LayerNorm(self.input_dim_primary)
        self.norm_op_2 = torch.nn.LayerNorm(self.input_dim_primary)

    def init_mlp(self):
        """
Initializes the MLP in the last step of the transformer."""
        if isinstance(self.mlp_structure, list):
            struc = max(self.mlp_structure)
        else:
            struc = self.mlp_structure
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_primary, struc),
            self.adn_fn(struc),
            torch.nn.Linear(struc, self.input_dim_primary),
        )

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
Forward pass.

        Args:
            X (torch.Tensor): tensor of shape [...,self.input_dim_primary]
            mask (torch.Tensor, optioanl): attention masking tensor. Should have
                shape []. Defaults to None.
            return_attention (bool, optional): also returns the attention.
                Defaults to False.

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
        """
        attention = self.mha(self.norm_op_1(X), mask=mask)
        X = X + self.drop_op_1(attention)
        X = X + self.drop_op_2(self.mlp(self.norm_op_2(X)))
        if return_attention is True:
            return X, attention
        return X


class SWINTransformerBlock(torch.nn.Module):
    """
Shifted window transformer module."""

    def __init__(
        self,
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        window_size: Size2dOr3d,
        in_channels: int,
        attention_dim: int = None,
        hidden_dim: int = None,
        embedding_size: int = None,
        shift_size: int = 0,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        dropout_rate_embedding: float = 0.0,
        embed_method: str = "linear",
        mlp_structure: Union[List[int], float] = [32, 32],
        use_pos_embed: bool = False,
        adn_fn=get_adn_fn(1, "identity", "gelu"),
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            window_size (Size2dOr3d, optional): window size for windowed
                multi-head attention. Defaults to None (no windowing)
            in_channels (int): number of channels in the input image.
            attention_dim (int): size of attention. Defaults to None (same as
                inferred input dimension).
            hidden_dim (int): size of hidden dimension (output of attention
                modules). Defaults to None (same as inferred input dimension).
            embedding_size (int, optional): size of the embedding. Defaults to
                None (same as inferred input dimension).
            shift_size (int, optional): size of shift in patches (will be
                multiplied by patch_size to get the actual patch_size).
                Defaults to 0 (no shift).
            dropout_rate (float, optional): dropout rate of the dropout
                operations applied throughout this module. Defaults to 0.0.
            dropout_rate_embedding (float, optional): dropout rate for the
                LinearEmbedding operation. Defaults to 0.0.
            embed_method (str, optional): . Defaults to "linear".
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (Union[List[int],float], optional): hidden layer
                structure. Should be a list of ints or float. If float, the
                structure becomes a single layer whose number of hidden units
                is the inferred input dimension scaled by mlp_strcture.
                Defaults to [32,32].
            use_pos_embed (bool, optional): includes positional embedding.
                Defaults to False.
            adn_fn (Callable, optional): function that returns a
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to
                get_adn_fn(1,"identity","gelu").
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.shift_size = shift_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.dropout_rate_embedding = dropout_rate_embedding
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.use_pos_embed = use_pos_embed
        self.adn_fn = adn_fn

        self.init_embedding()
        self.init_drop_ops()
        self.init_layers()
        self.init_mask_if_necessary()

    def init_embedding(self):
        self.embedding = LinearEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            window_size=self.window_size,
            dropout_rate=self.dropout_rate_embedding,
            embed_method=self.embed_method,
            out_dim=self.embedding_size,
            use_pos_embed=self.use_pos_embed,
            channels_last=True,
        )
        self.input_dim_primary = self.embedding.true_n_features

    def init_mask_if_necessary(self):
        # window_size here has to be given in *number of patches*
        self.attention_mask = generate_mask(
            image_size=[
                x // y for x, y in zip(self.image_size, self.patch_size)
            ],
            window_size=[
                x // y for x, y in zip(self.window_size, self.patch_size)
            ],
            shift_size=self.shift_size,
        )

    def init_layers(self):
        if isinstance(self.mlp_structure, float):
            self.mlp_structure = [int(self.in_channels * self.mlp_structure)]

        input_dim_primary = self.input_dim_primary

        if self.hidden_dim is None:
            hidden_dim = input_dim_primary
        else:
            hidden_dim = self.hidden_dim

        if self.attention_dim is None:
            attention_dim = input_dim_primary
        else:
            attention_dim = self.attention_dim

        self.mha = MultiHeadSelfAttention(
            input_dim_primary,
            attention_dim,
            hidden_dim,
            input_dim_primary,
            window_size=self.window_size,
            dropout_rate=self.dropout_rate,
            n_heads=self.n_heads,
        )

        self.norm_op_1 = torch.nn.LayerNorm(input_dim_primary)
        self.norm_op_2 = torch.nn.LayerNorm(self.in_channels)
        self.mlp = MLP(
            self.in_channels,
            self.in_channels,
            self.mlp_structure,
            self.adn_fn,
        )

    def init_drop_ops(self):
        """
Initializes the dropout operations."""
        self.drop_op_1 = torch.nn.Dropout(self.dropout_rate)
        self.drop_op_2 = torch.nn.Dropout(self.dropout_rate)

    def apply_shift_and_attention(self, X: torch.Tensor):
        cyc = [-s * self.shift_size for s in self.patch_size]
        rev_cyc = [-c for c in cyc]
        X = cyclic_shift_batch(X, cyc)
        X = self.embedding(X)
        attention = self.mha(self.norm_op_1(X), mask=self.attention_mask)
        attention = self.embedding.rearrange_inverse_basic(attention)
        attention = cyclic_shift_batch(attention, rev_cyc)
        attention = self.embedding.rearrange(attention)
        return attention

    def roll(self, X: torch.Tensor, shifts: List[int]):
        """
Rolls a tensor along the last dimensions.

        Args:
            X (torch.Tensor): tensor to roll.
            shifts (List[int]): list of shifts to apply.

        Returns:
            torch.Tensor: rolled tensor.
        """
        dims = [i + 1 for i in range(1, len(shifts) + 1)]
        return torch.roll(X, shifts=shifts, dims=dims)

    def forward(
        self, X: torch.Tensor, scale: int = None
    ) -> Tuple[torch.Tensor, TensorList]:
        """
Forward pass, based on MONAI implementation [1].

        [1] https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]
            scale (int): downsampling scale for output. Defaults to None
                (returns the non-rearranged output).

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        X = move_axis(X, 1, -1)
        shortcut = X
        if scale is not None:
            if isinstance(scale, int):
                scale = [scale for _ in self.patch_size]
        ss = self.shift_size
        if isinstance(ss, int):
            ss = [ss] * len(self.patch_size)
        if any(i > 0 for i in ss):
            shifted_X = self.roll(X, shifts=[-s for s in ss])
        else:
            shifted_X = X
        embedded_X = self.embedding(shifted_X)
        attention = self.mha(
            self.norm_op_1(embedded_X), mask=self.attention_mask
        )
        if self.embedding.embed_method == "convolutional":
            shifted_X = self.embedding.rearrange_inverse(attention)
        else:
            shifted_X = self.embedding.rearrange_inverse_basic(attention)
        if any(i > 0 for i in ss):
            X = self.roll(shifted_X, shifts=ss)
        else:
            X = shifted_X

        X = shortcut + self.drop_op_1(shifted_X)
        X = X + self.drop_op_2(self.mlp(self.norm_op_2(X)))
        X = move_axis(X, -1, 1)
        if scale is not None:
            X = einops_rescale(X, scale)
        return X

    def forward_old(
        self, X: torch.Tensor, scale: int = None
    ) -> Tuple[torch.Tensor, TensorList]:
        """
Forward pass.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]
            scale (int): downsampling scale for output. Defaults to None
                (returns the non-rearranged output).

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        if self.shift_size > 0:
            attention = self.apply_shift_and_attention(X)
            X = self.embedding(X)
        else:
            X = self.embedding(X)
            attention = self.mha(self.norm_op_1(X), mask=None)
        X = X + self.drop_op_1(attention)
        X = X + self.drop_op_2(self.mlp(self.norm_op_2(X)))

        X = self.embedding.rearrange_rescale(X, scale)
        return X


class TransformerBlockStack(torch.nn.Module):
    """
Convenience function that stacks a series of transformer blocks."""

    def __init__(
        self,
        number_of_blocks: int,
        input_dim_primary: int,
        attention_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        mlp_structure: List[int] = [128],
        dropout_rate: float = 0.0,
        adn_fn: Callable = get_adn_fn(1, "identity", "gelu"),
        window_size: Size2dOr3d = None,
    ):
        """
        Args:
            number_of_blocks (int): number of blocks to be stacked.
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (List[int], optional): hidden layer structure.
                Should be a list of ints. Defaults to [128].
            dropout_rate (float, optional): dropout rate, applied to the output
                of each sub-layer. Defaults to 0.0.
            adn_fn (Callable, optional): function that returns a
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to
                get_adn_fn(1,"identity","gelu").
            window_size (bool, optional): window_size for windowed W-MSA.
                Defaults to None (regular block of transformers).
        """
        super().__init__()
        self.number_of_blocks = number_of_blocks
        self.input_dim_primary = input_dim_primary
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_structure = mlp_structure
        self.dropout_rate = dropout_rate
        self.adn_fn = adn_fn
        self.window_size = window_size

        self.init_transformer_blocks()

    def check_mlp_structure(
        self, x: Union[List[List[int]], List[int]]
    ) -> List[List[int]]:
        """
Checks and corrects the MLP structure to see that everything is
        as expected.

        Args:
            x (Union[List[List[int]], List[int]]): MLP structure or list of
                MLP structures.

        Returns:
            List[List[int]]: list of MLP structures.
        """
        if isinstance(x, list) is False:
            x = [x]
        if isinstance(x[0], list) is False:
            return [x for _ in range(self.number_of_blocks)]
        else:
            return x

    def convert_to_list_if_necessary(
        self, x: Union[List[int], int]
    ) -> List[int]:
        """
Checks and corrects inputs if necessary.

        Args:
            x (Union[List[int],int]): integer or list of integers

        Returns:
            List[int]: list of integers.
        """
        if isinstance(x, list) is False:
            x = [x for _ in range(self.number_of_blocks)]
        else:
            x = x
        self.check_if_consistent(x)
        return x

    def check_if_consistent(self, x: Sequence):
        """
Checks that the size of x is self.number_of_blocks

        Args:
            x (_type_): _description_
        """
        assert len(x) == self.number_of_blocks

    def init_transformer_blocks(self):
        """
Initialises the transformer blocks."""
        input_dim_primary = self.convert_to_list_if_necessary(
            self.input_dim_primary
        )
        attention_dim = self.convert_to_list_if_necessary(self.attention_dim)
        hidden_dim = self.convert_to_list_if_necessary(self.hidden_dim)
        n_heads = self.convert_to_list_if_necessary(self.n_heads)
        mlp_structure = self.check_mlp_structure(self.mlp_structure)
        self.transformer_blocks = torch.nn.ModuleList([])
        for i, a, h, n, m in zip(
            input_dim_primary,
            attention_dim,
            hidden_dim,
            n_heads,
            mlp_structure,
        ):
            self.transformer_blocks.append(
                TransformerBlock(
                    input_dim_primary=i,
                    attention_dim=a,
                    hidden_dim=h,
                    n_heads=n,
                    mlp_structure=m,
                    dropout_rate=self.dropout_rate,
                    window_size=self.window_size,
                    adn_fn=self.adn_fn,
                )
            )

    def forward(
        self,
        X: torch.Tensor,
        return_at: Union[str, List[int]] = "end",
        return_attention: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, TensorList],
        Tuple[torch.Tensor, TensorList, TensorList],
    ]:
        """
Forward pass.

        Args:
            X (torch.Tensor): tensor of shape [...,self.input_dim_primary]
            return_at (Union[str,List[int]], optional): sets the intermediary
                outputs that will be returned together with the final output.
            return_attention (bool, optional): also returns the attention for
                all. blocks in return_at. Defaults to False.

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith attention outputs, where i is contained in return_at.
        """
        if isinstance(return_at, list):
            assert (
                max(return_at) < self.number_of_blocks
            ), "max(return_at) should be smaller than self.number_of_blocks"
        if return_at == "end" or return_at is None:
            return_at = []
        outputs = []
        attentions = []
        for i, block in enumerate(self.transformer_blocks):
            X = block(X, return_attention=return_attention)
            if return_attention is True:
                X, attention = X
            if i in return_at:
                outputs.append(X)
                attentions.append(attention)
        if return_attention is True:
            if return_at == []:
                return X, outputs, [attention]
            else:
                return X, outputs, attentions
        return X, outputs


class SWINTransformerBlockStack(torch.nn.Module):
    """
Shifted window transformer stack module."""

    def __init__(
        self,
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        window_size: Size2dOr3d,
        shift_sizes: List[int],
        in_channels: int,
        attention_dim: int = None,
        hidden_dim: int = None,
        embedding_size: int = None,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        dropout_rate_embedding: float = 0.0,
        embed_method: str = "linear",
        mlp_structure: Union[List[int], float] = [128],
        use_pos_embed: bool = False,
        adn_fn=get_adn_fn(1, "identity", "gelu"),
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            window_size (Size2dOr3d, optional): window size for windowed
                multi-head attention. Defaults to None (no windowing)
            shift_sizes (List[int], optional): list of shift sizes in
                patches.
            in_channels (int): number of channels in the input image.
            input_dim_primary (int): size of input.
            attention_dim (int): size of attention.
            hidden_dim (int): size of hidden dimension (output of attention
                modules).
            embedding_size (int, optional): size of the embedding. Defaults to
                None (same as inferred output dimension).
            dropout_rate (float, optional): dropout rate of the dropout
                operations applied throughout this module. Defaults to 0.0.
            dropout_rate_embedding (float, optional): dropout rate of the
                LinearEmbedding module. Defaults to 0.0.
            embed_method (str, optional): . Defaults to "linear".
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (Union[List[int],float], optional): hidden layer
                structure. Should be a list of ints or float. If float, the
                structure becomes a single layer whose number of hidden units
                is the inferred input dimension scaled by mlp_strcture.
                Defaults to [32,32].
            use_pos_embed (bool, optional): includes positional embedding.
                Defaults to False.
            adn_fn (Callable, optional): function that returns a
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to
                get_adn_fn(1,"identity","gelu").
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_sizes = shift_sizes
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.dropout_rate_embedding = dropout_rate_embedding
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.use_pos_embed = use_pos_embed
        self.adn_fn = adn_fn

        self.init_swin_transformers()

    def convert_mlp_structure(
        self, x: Union[List[List[int]], List[int], float]
    ) -> List[List[int]]:
        """
Checks and corrects list structure to see that everything is
        as expected.

        Args:
            x (Union[List[List[int]], List[int]]): list structure or list of
                lists structure.

        Returns:
            List[List[int]]: list of lists structures.
        """
        if isinstance(x, float) is True:
            return [x for _ in self.shift_sizes]
        if isinstance(x[0], list) is False:
            return [x for _ in self.shift_sizes]
        else:
            return x

    def convert_to_list_if_necessary(
        self, x: Union[List[int], int]
    ) -> List[int]:
        """
Checks and corrects inputs if necessary.

        Args:
            x (Union[List[int],int]): integer or list of integers

        Returns:
            List[int]: list of integers.
        """
        if isinstance(x, list) is False:
            return [x for _ in self.shift_sizes]
        else:
            return self.check_if_consistent(x)

    def check_if_consistent(self, x: Sequence):
        """
Checks that the size of x is self.number_of_blocks

        Args:
            x (Sequence): _description_
        """
        assert len(x) == len(self.shift_sizes)

    def init_swin_transformers(self):
        attention_dim = self.convert_to_list_if_necessary(self.attention_dim)
        hidden_dim = self.convert_to_list_if_necessary(self.hidden_dim)
        n_heads = self.convert_to_list_if_necessary(self.n_heads)
        dropout_rate = self.convert_to_list_if_necessary(self.dropout_rate)
        mlp_structure = self.convert_mlp_structure(self.mlp_structure)
        self.stbs = torch.nn.ModuleList([])
        first = True
        for ss, ad, hd, nh, dr, mlp_s in zip(
            self.shift_sizes,
            attention_dim,
            hidden_dim,
            n_heads,
            dropout_rate,
            mlp_structure,
        ):
            stb = SWINTransformerBlock(
                image_size=self.image_size,
                patch_size=self.patch_size,
                window_size=self.window_size,
                in_channels=self.in_channels,
                attention_dim=ad,
                hidden_dim=hd,
                embedding_size=self.embedding_size,
                shift_size=ss,
                n_heads=nh,
                dropout_rate_embedding=self.dropout_rate_embedding,
                dropout_rate=dr,
                embed_method=self.embed_method if first else "linear",
                mlp_structure=mlp_s,
                adn_fn=self.adn_fn,
                use_pos_embed=first and self.use_pos_embed,
            )
            self.stbs.append(stb)
            if first:
                first = False

    def forward(
        self, X: torch.Tensor, scale: int = 1
    ) -> Tuple[torch.Tensor, TensorList]:
        """
Forward pass.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        for block in self.stbs[:-1]:
            X = block(X, scale=1)
        X = self.stbs[-1](X, scale=scale)
        return X


class ViT(torch.nn.Module):
    """
Vision transformer module. Put more simply, it is the
    concatenation of a LinearEmbedding and a TransformberBlockStack [1]. This
    model includes the possibility of PatchErasing regularization [2].

    [1] https://arxiv.org/abs/2010.11929
    [2] https://arxiv.org/abs/2209.15006
    """

    def __init__(
        self,
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        in_channels: int,
        number_of_blocks: int,
        attention_dim: int,
        hidden_dim: int = None,
        embedding_size: int = None,
        window_size: Size2dOr3d = None,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        use_pos_embed: bool = True,
        embed_method: str = "linear",
        mlp_structure: Union[List[int], float] = [128],
        adn_fn=get_adn_fn(1, "identity", "gelu"),
        use_class_token: bool = False,
        n_registers: int = 0,
        learnable_embedding: bool = True,
        channel_to_token: bool = False,
        patch_erasing: float = None,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            in_channels (int): number of channels in the input image.
            number_of_blocks (int): number of blocks to be stacked.
            attention_dim (int): size of attention.
            hidden_dim (int, optional): size of hidden dimension (output of
                attention modules). Defaults to None (same as inferred input
                dimension).
            embedding_size (int, optional): size of the embedding. Defaults to
                None (same as inferred output dimension).
            dropout_rate (float, optional): dropout rate of the dropout
                operations applied throughout this module. Defaults to 0.0.
            use_pos_embed (bool, optional): includes positional embedding.
                Defaults to True.
            embed_method (str, optional): image embedding method. Defaults to
                "linear".
            window_size (Size2dOr3d, optional): window size for windowed MSA.
                Defaults to None (regular ViT).
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (Union[List[int],float], optional): hidden layer
                structure. Should be a list of ints or float. If float, the
                structure becomes a single layer whose number of hidden units
                is the inferred input dimension scaled by mlp_strcture.
                Defaults to [32,32].
            adn_fn (Callable, optional): function that returns a
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to
                get_adn_fn(1,"identity","gelu").
            use_class_token (bool, optional): adds classification token to
                embedding layer. Defaults to False.
            n_registers (int, optional): number of registers to use as
                suggested in Darcet (2024). Defaults to 0.
            learnable_embedding (bool, optional): if embedding is
                non-trainable, a sinusoidal positional embedding is used.
                Defaults to True.
            channel_to_token (bool, optional): embeds the channel dimension as
                tokens rather than as part of the embedding. Works only with
                "linear" embedding. Defaults to False.
            patch_erasing (float, optional): erases (sets to 0) a fraction of
                patches. Defaults to None (no patch erasing).
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.number_of_blocks = number_of_blocks
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.use_pos_embed = use_pos_embed
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn = adn_fn
        self.use_class_token = use_class_token
        self.n_registers = n_registers
        self.learnable_embedding = learnable_embedding
        self.channel_to_token = channel_to_token
        self.patch_erasing = patch_erasing

        self.embedding = LinearEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            window_size=self.window_size,
            out_dim=embedding_size,
            embed_method=self.embed_method,
            use_pos_embed=self.use_pos_embed,
            dropout_rate=self.dropout_rate,
            use_class_token=self.use_class_token,
            n_registers=self.n_registers,
            learnable_embedding=self.learnable_embedding,
            channel_to_token=self.channel_to_token,
        )
        self.input_dim_primary = self.embedding.true_n_features
        if self.patch_erasing is None:
            self.patch_erasing_op = None
        elif callable(self.patch_erasing):
            self.patch_erasing_op = self.patch_erasing
        else:
            self.patch_erasing_op = ChannelDropout(self.patch_erasing)

        if isinstance(self.mlp_structure, float):
            self.mlp_structure = [
                int(self.input_dim_primary * self.mlp_structure)
            ]

        if embedding_size is not None:
            input_dim_primary = embedding_size
        else:
            input_dim_primary = self.input_dim_primary

        if self.hidden_dim is None:
            hidden_dim = input_dim_primary
        else:
            hidden_dim = self.hidden_dim

        if self.attention_dim is None:
            attention_dim = input_dim_primary
        else:
            attention_dim = self.attention_dim

        self.tbs = TransformerBlockStack(
            number_of_blocks=self.number_of_blocks,
            input_dim_primary=input_dim_primary,
            attention_dim=attention_dim,
            hidden_dim=hidden_dim,
            n_heads=self.n_heads,
            mlp_structure=self.mlp_structure,
            dropout_rate=self.dropout_rate,
            adn_fn=self.adn_fn,
            window_size=window_size,
        )

    def forward(
        self, X: torch.Tensor, return_at: Union[str, List[int]] = "end"
    ) -> Tuple[torch.Tensor, TensorList]:
        """
Forward pass.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]
            return_at (Union[str,List[int]], optional): sets the intermediary
                outputs that will be returned together with the final output.

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        if isinstance(return_at, list):
            assert (
                max(return_at) < self.number_of_blocks
            ), "max(return_at) should be smaller than self.number_of_blocks"
        embeded_X = self.embedding(X)
        if self.patch_erasing is not None:
            embeded_X = self.patch_erasing_op(embeded_X)
        if return_at == "end" or return_at is None:
            return_at = []
        outputs = []
        for i, block in enumerate(self.tbs.transformer_blocks):
            embeded_X = block(embeded_X)
            if i in return_at:
                outputs.append(embeded_X)
        return embeded_X, outputs


class FactorizedViT(torch.nn.Module):
    """
Factorized vision transformer module. Put more simply, it is the
    concatenation of a SliceLinearEmbedding and two TransformberBlockStack [1]
    (corresponding to within and between slice interactions). This model
    includes the possibility of PatchErasing regularization [2].

    [1] https://www.sciencedirect.com/science/article/pii/S0010482522008459?via%3Dihub
    [2] https://arxiv.org/abs/2209.15006
    """

    def __init__(
        self,
        image_size: Size2dOr3d,
        patch_size: Size2dOr3d,
        in_channels: int,
        number_of_blocks: int,
        attention_dim: int,
        hidden_dim: int = None,
        embedding_size: int = None,
        n_heads: int = 4,
        dropout_rate: float = 0.0,
        embed_method: str = "linear",
        mlp_structure: Union[List[int], float] = [128],
        adn_fn=get_adn_fn(1, "identity", "gelu"),
        use_class_token: bool = False,
        n_registers: int = 0,
        learnable_embedding: bool = True,
        patch_erasing: float = None,
    ):
        """
        Args:
            image_size (Size2dOr3d): size of the input image.
            patch_size (Size2dOr3d): size of the patch size.
            in_channels (int): number of channels in the input image.
            number_of_blocks (int): number of blocks to be stacked.
            attention_dim (int): size of attention.
            hidden_dim (int, optional): size of hidden dimension (output of
                attention modules). Defaults to None (same as inferred input
                dimension).
            embedding_size (int, optional): size of the embedding. Defaults to
                None (same as inferred output dimension).
            dropout_rate (float, optional): dropout rate of the dropout
                operations applied throughout this module. Defaults to 0.0.
            embed_method (str, optional): image embedding method. Defaults to
                "linear".
            window_size (Size2dOr3d, optional): window size for windowed MSA.
                Defaults to None (regular ViT).
            n_heads (int, optional): number of attention heads. Defaults to 4.
            mlp_structure (Union[List[int],float], optional): hidden layer
                structure. Should be a list of ints or float. If float, the
                structure becomes a single layer whose number of hidden units
                is the inferred input dimension scaled by mlp_strcture.
                Defaults to [32,32].
            adn_fn (Callable, optional): function that returns a
                torch.nn.Module that does activation/dropout/normalization,
                used in the MLP sub-layer. Should take as arguments the number
                of channels in a given layer. Defaults to
                get_adn_fn(1,"identity","gelu").
            use_class_token (bool, optional): adds classification token to
                embedding layer. Defaults to False.
            n_registers (int, optional): number of registers to use as
                suggested in Darcet (2024). Defaults to 0.
            learnable_embedding (bool, optional): if embedding is
                non-trainable, a sinusoidal positional embedding is used.
                Defaults to True.
            patch_erasing (float, optional): erases (sets to 0) a fraction of
                patches. Defaults to None (no patch erasing).
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.number_of_blocks = number_of_blocks
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.embed_method = embed_method
        self.mlp_structure = mlp_structure
        self.adn_fn = adn_fn
        self.use_class_token = use_class_token
        self.n_registers = n_registers
        self.learnable_embedding = learnable_embedding
        self.patch_erasing = patch_erasing

        self.embedding = SliceLinearEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embedding_size=self.embedding_size,
            dropout_rate=self.dropout_rate,
            embed_method=self.embed_method,
            use_class_token=self.use_class_token,
            n_registers=self.n_registers,
            learnable_embedding=self.learnable_embedding,
            out_dim=self.embedding_size,
        )
        self.input_dim_primary = self.embedding.true_n_features
        input_dim_primary = self.input_dim_primary
        if self.patch_erasing:
            self.patch_erasing_op = ChannelDropout(self.patch_erasing, 2)
        else:
            self.patch_erasing_op = None

        if self.hidden_dim is None:
            hidden_dim = input_dim_primary
        else:
            hidden_dim = self.hidden_dim

        if self.attention_dim is None:
            attention_dim = input_dim_primary
        else:
            attention_dim = self.attention_dim

        a = self.number_of_blocks // 2
        b = self.number_of_blocks - a
        self.transformer_block_within = TransformerBlockStack(
            number_of_blocks=b,
            input_dim_primary=input_dim_primary,
            attention_dim=attention_dim,
            hidden_dim=hidden_dim,
            n_heads=self.n_heads,
            mlp_structure=self.mlp_structure,
            dropout_rate=self.dropout_rate,
            adn_fn=self.adn_fn,
        )

        self.transformer_block_between = TransformerBlockStack(
            number_of_blocks=a,
            input_dim_primary=input_dim_primary,
            attention_dim=attention_dim,
            hidden_dim=hidden_dim,
            n_heads=self.n_heads,
            mlp_structure=self.mlp_structure,
            dropout_rate=self.dropout_rate,
            adn_fn=self.adn_fn,
        )

        if self.use_class_token is True:
            self.slice_class_token = torch.nn.Parameter(
                torch.zeros([1, 1, input_dim_primary])
            )

        if self.n_registers > 0:
            self.slice_registers = torch.nn.Parameter(
                torch.zeros([1, self.n_registers, input_dim_primary])
            )

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, TensorList]:
        """
Forward pass.

        Args:
            X (torch.Tensor): tensor of shape
                [-1,self.in_channels,*self.image_size]

        Returns:
            torch.Tensor: tensor of shape [...,self.input_dim_primary]
            List[torch.Tensor]: list of intermediary tensors corresponding to
                the ith transformer outputs, where i is contained in return_at.
                Same shape as the final output.
        """
        embeded_X = self.embedding(X)
        if self.patch_erasing_op is not None:
            embeded_X = self.patch_erasing_op(embeded_X)
        embeded_X, _ = self.transformer_block_within(embeded_X)
        if self.n_registers > 0:
            embeded_X = embeded_X[:, :, self.n_registers :]
            registers = einops.repeat(
                self.slice_registers, "() n e -> b n e", b=X.shape[0]
            )
            embeded_X = torch.concat([registers, embeded_X], 1)
        if self.use_class_token is True:
            embeded_X = embeded_X[:, :, 0]
            class_token = einops.repeat(
                self.slice_class_token, "() n e -> b n e", b=X.shape[0]
            )
            embeded_X = torch.concat([class_token, embeded_X], 1)
        else:
            embeded_X = embeded_X.mean(-2)
            embeded_X = embeded_X
        embeded_X, _ = self.transformer_block_between(embeded_X)
        return embeded_X
