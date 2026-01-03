# import gc
# import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
from torch import nn, Tensor

# import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .norm import create_norm, create_activation
from .FlashMultiHead import FlashMHA


class FlashTransformerLayer(nn.Module):
    r"""FlashTransformerLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=128,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        norm="layernorm",  # batchnorm, layernorm, rmsnorm
        norm_first=True,
        batch_first=True,
        flash=False,
        device=None,
        dtype=None,
    ) -> None:

        super().__init__()

        self.flash = flash
        self.device = device
        self.dtype = dtype
        self.norm = norm

        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            dropout=dropout,
            causal=False,
            device=device,
            dtype=dtype,
            flash=flash,
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        # print(factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first

        self.norm1 = create_norm(norm, d_model)
        self.norm2 = create_norm(norm, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = create_activation(activation)

    # 自注意力 block
    def _sa_block(self, x, need_weights=False):
        x, attn = self.self_attn(x, x, x, need_weights=need_weights)
        return self.dropout1(x), attn

    # 前馈网络 block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x, output_attentions=False):
        if self.norm_first:
            x_prime, attn = self._sa_block(
                self.norm1(x), need_weights=output_attentions
            )
            x = x + x_prime
            x = x + self._ff_block(self.norm2(x))
        else:
            x_prime, attn = self._sa_block(x, need_weights=output_attentions)
            x = self.norm1(x + x_prime)
            x = self.norm2(x + self._ff_block(x))

        return x, attn
