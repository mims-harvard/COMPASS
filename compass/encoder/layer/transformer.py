import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import MultiheadAttention, Dropout, LayerNorm
from typing import Optional, Callable, Union
from abc import ABC, abstractmethod


# 自定义线性层，包括谱归一化
class Linear(nn.Linear):
    # 初始化函数
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        # 初始化谱归一化
        self.register_buffer(
            "u", nn.functional.normalize(torch.randn(in_features), dim=0)
        )
        sigma = self.get_sigma()
        self.register_buffer("spectral_norm", sigma)
        self.sigma = nn.Parameter(torch.ones(1))

    # 获取谱归一化的 sigma 值
    def get_sigma(self):
        with torch.no_grad():
            u = self.u
            v = self.weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = self.weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            self.u.data.copy_(u)
        return torch.einsum("c,cd,d->", v, self.weight, u)

    # 重写 forward 方法，应用谱归一化
    def forward(self, x):
        weight = (self.sigma / self.spectral_norm) * self.weight
        return nn.functional.linear(x, weight, self.bias)


# 获取激活函数
def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


# 抽象 Transformer 层基类
class AbstractTrasnformerLayer(ABC):
    @abstractmethod
    def __init__(
        self, embed_dim, num_heads, dropout, norm, norm_first: bool, causal: bool
    ):
        pass

    @abstractmethod
    def forward(self, x, attn_mask, output_attentions):
        pass


# 标准 Transformer 编码器层
class TransformerEncoderLayer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = (
            _get_activation_fn(activation)
            if isinstance(activation, str)
            else activation
        )

    # 前向传播函数
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # 自注意力 block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        return self.dropout1(x)

    # 前馈网络 block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Vanilla Transformer 层
class VanillaTransformerLayer(nn.Module, AbstractTrasnformerLayer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        norm="layernorm",
        norm_first=True,
        causal=False,
    ):
        super().__init__()
        assert norm == "layernorm", "Vanilla transformer only supports layernorm."
        assert not causal, "Vanilla transformer does not support causal inference."
        self.layer = TransformerEncoderLayer(
            embed_dim,
            num_heads,
            embed_dim * 2,
            dropout,
            activation="gelu",
            norm_first=norm_first,
        )
        self.support_output_attentions = False

    def forward(self, x, attn_mask=None, output_attentions=False):
        assert (
            not output_attentions
        ), "output_attentions not implemented for VanillaTransformer"
        return self.layer(x, attn_mask)
