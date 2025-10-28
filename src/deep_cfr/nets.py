"""
Various neural network building blocks and flexible network class.

The FlexibleNet class can be configured as either:
  - a pure MLP
  - a Conv2D feature extractor followed by an MLP head

Examples usages:

# Flattened MLP for regret/policy nets:

regret_net = FlexibleNet(
    mode="mlp",
    input_shape=(160,),
    output_dim=40,
    mlp_hidden=[512, 256],
    mlp_act="relu",
    mlp_norm="none",
    mlp_dropout=0.0,
    mlp_residual=False,
)

policy_net = FlexibleNet(
    mode="mlp",
    input_shape=(160,),
    output_dim=40,
    mlp_hidden=[512, 256],
    mlp_act="relu",
    mlp_norm="none",
    mlp_dropout=0.1,
)
# Forward:
# obs_np: (B, 4, 40) -> torch (B, 160)
# logits/advantages = net(obs_flat)

# A classic CNN:

atari_net = FlexibleNet(
    mode="conv2d_mlp",
    input_shape=(4, 84, 84),
    output_dim=6,
    conv_channels=[32, 64, 64],
    conv_kernels=[8, 4, 3],
    conv_strides=[4, 2, 1],
    conv_paddings=[0, 0, 0],
    mlp_hidden=[512],
    mlp_act="relu",
)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

# ---- small registry ----
_ACTS: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "lrelu": lambda: nn.LeakyReLU(0.1),
    "none": nn.Identity,
}
_NORM_2D: Dict[str, callable] = {
    "batch": nn.BatchNorm2d,
    "layer": lambda c: nn.GroupNorm(1, c),  # LayerNorm alternative for 2D
    "none": lambda c: nn.Identity(),
}
_NORM_1D: Dict[str, callable] = {
    "batch": nn.BatchNorm1d,
    "layer": lambda c: nn.LayerNorm(c),
    "none": lambda c: nn.Identity(),
}


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Softmax over legal actions only. logits, mask: (B, A). mask in {0,1}.

    :param logits: torch.Tensor logits (B, A)
    :param mask: torch.Tensor legal action mask (B, A)
    :param eps: small value to avoid division by zero
    :return: torch.Tensor probabilities (B, A)
    """
    very_neg = torch.finfo(logits.dtype).min / 2
    masked = torch.where(mask > 0, logits, very_neg)
    probs = torch.softmax(masked, dim=-1)
    # Renormalize in case mask is all zeros (fallback to uniform)
    z = (probs * mask).sum(dim=-1, keepdim=True).clamp_min(eps)
    return (probs * mask) / z


def positive_regret_policy(
    advantages: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    σ(a) ∝ max(0, A(a)) over legal actions (uniform if all non-positive).
    advantages, mask: (B, A)

    :param advantages: torch.Tensor advantages/regrets (B, A)
    :param mask: torch.Tensor legal action mask (B, A)
    :param eps: small value to avoid division by zero
    :return: torch.Tensor probabilities (B, A)
    """
    pos = torch.clamp(advantages, min=0.0) * mask
    z = pos.sum(dim=-1, keepdim=True)
    # If all <=0, use uniform over legal actions
    uniform = mask / mask.sum(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(z > eps, pos / z, uniform)


class ConvBlock2D(nn.Module):
    """
    A 2D convolutional block with optional normalization, activation, dropout, and residual connection.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param k: Kernel size.
    :param s: Stride.
    :param p: Padding.
    :param act: Activation function name.
    :param norm: Normalization type.
    :param dropout2d: Dropout probability.
    :param residual: Whether to include a residual connection.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        act: str = "relu",
        norm: str = "none",
        dropout2d: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.norm = _NORM_2D[norm](out_ch)
        self.act = _ACTS[act]()
        self.drop = nn.Dropout2d(dropout2d) if dropout2d > 0 else nn.Identity()
        self.residual = residual and (in_ch == out_ch and s == 1)
        if residual and not self.residual:
            # incompatible shapes for residual; silently disable
            self.residual = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        if self.residual:
            y = y + x
        return y


class MLPBlock(nn.Module):
    """
    A fully connected block with optional normalization, activation, dropout, and residual connection.

    :param in_dim: Input feature dimension.
    :param out_dim: Output feature dimension.
    :param act: Activation function name.
    :param norm: Normalization type.
    :param dropout: Dropout probability.
    :param residual: Whether to include a residual connection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: str = "relu",
        norm: str = "none",
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = _NORM_1D[norm](out_dim)
        self.act = _ACTS[act]()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual and (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        if self.residual:
            y = y + x
        return y


class FlexibleNet(nn.Module):
    """
    A configurable network that can be:
      - 'mlp': pure MLP
      - 'conv2d_mlp': Conv2D feature extractor followed by MLP head

    You specify layer shapes with lists; the class infers the flatten size.

    :param mode: Network mode, either 'mlp' or 'conv2d_mlp'.
    :param input_shape: Input shape tuple. For 'mlp', (D,). For 'conv2d_mlp', (C,H,W).
    :param output_dim: Output dimension (e.g., number of actions).
    :param conv_channels: List of output channels for each Conv2D layer (for 'conv2d_mlp').
    :param conv_kernels: List of kernel sizes for each Conv2D layer (for 'conv2d_mlp').
    """

    def __init__(
        self,
        mode: Literal["mlp", "conv2d_mlp"],
        input_shape: Tuple[int, ...],  # (C,H,W) for conv2d_mlp, or (D,) for mlp
        output_dim: int,  # e.g., num_actions
        # Conv config
        conv_channels: Optional[List[int]] = None,
        conv_kernels: Optional[List[int]] = None,
        conv_strides: Optional[List[int]] = None,
        conv_paddings: Optional[List[int]] = None,
        conv_act: str = "relu",
        conv_norm: str = "none",
        conv_dropout2d: float = 0.0,
        conv_residual: bool = False,
        # MLP head config
        mlp_hidden: Optional[List[int]] = None,
        mlp_act: str = "relu",
        mlp_norm: str = "none",
        mlp_dropout: float = 0.0,
        mlp_residual: bool = False,
    ):
        super().__init__()
        self.mode = mode

        if mode == "mlp":
            assert len(input_shape) == 1, "For 'mlp', input_shape must be (D,)."
            in_dim = input_shape[0]
            self.backbone = self._build_mlp(
                in_dim, mlp_hidden or [], mlp_act, mlp_norm, mlp_dropout, mlp_residual
            )
            last = (mlp_hidden or [in_dim])[-1] if mlp_hidden else in_dim
            self.head = nn.Linear(last, output_dim)

        elif mode == "conv2d_mlp":
            assert (
                len(input_shape) == 3
            ), "For 'conv2d_mlp', input_shape must be (C,H,W)."
            C, H, W = input_shape
            # Build conv stack
            conv_channels = conv_channels or [32, 64, 64]
            conv_kernels = conv_kernels or [3, 3, 3]
            conv_strides = conv_strides or [1, 2, 2]
            conv_paddings = conv_paddings or [1, 1, 1]
            assert (
                len(
                    {
                        len(conv_channels),
                        len(conv_kernels),
                        len(conv_strides),
                        len(conv_paddings),
                    }
                )
                == 1
            ), "Conv parameter lists must have the same length."

            conv_layers: List[nn.Module] = []
            in_ch = C
            for out_ch, k, s, p in zip(
                conv_channels, conv_kernels, conv_strides, conv_paddings
            ):
                conv_layers.append(
                    ConvBlock2D(
                        in_ch,
                        out_ch,
                        k,
                        s,
                        p,
                        act=conv_act,
                        norm=conv_norm,
                        dropout2d=conv_dropout2d,
                        residual=conv_residual,
                    )
                )
                in_ch = out_ch
            self.conv = nn.Sequential(*conv_layers)

            # Infer flatten dim
            with torch.no_grad():
                dummy = torch.zeros(1, C, H, W)
                z = self.conv(dummy)
                flat_dim = z.view(1, -1).shape[1]

            # Build MLP head (including final Linear)
            self.backbone = nn.Identity()
            self.mlp = self._build_mlp(
                flat_dim, mlp_hidden or [], mlp_act, mlp_norm, mlp_dropout, mlp_residual
            )
            last = (mlp_hidden or [flat_dim])[-1] if mlp_hidden else flat_dim
            self.head = nn.Linear(last, output_dim)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _build_mlp(
        in_dim: int,
        hidden: List[int],
        act: str,
        norm: str,
        dropout: float,
        residual: bool,
    ) -> nn.Sequential:
        """
        Build an MLP from a list of hidden layer sizes.
        :param in_dim: Input feature dimension.
        :param hidden: List of hidden layer sizes.
        :param act: Activation function name.
        :param norm: Normalization type.
        :param dropout: Dropout probability.
        :param residual: Whether to include residual connections.
        :return: nn.Sequential MLP module.
        """
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(
                MLPBlock(
                    last, h, act=act, norm=norm, dropout=dropout, residual=residual
                )
            )
            last = h
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp":
            # x: (B, D)
            z = self.backbone(x)
            out = self.head(z)
            return out
        else:
            # conv2d_mlp: x: (B, C, H, W)
            z = self.conv(x)
            z = z.flatten(start_dim=1)
            z = self.mlp(z)
            out = self.head(z)
            return out
