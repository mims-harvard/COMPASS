import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
from .norm import create_norm
from .transformer import AbstractTrasnformerLayer


class CosformerAttention(nn.Module):
    """
    cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout_rate=0.0,
        causal=False,
        has_outproj=True,
        act_fun="gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if kdim is not None else embed_dim
        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # outprojection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # dropout rate
        self.dropout_rate = dropout_rate
        self.attn_dropout = nn.Dropout(dropout_rate)
        # causal
        self.causal = causal

        assert (
            self.embed_dim % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        elif act_fun == "gelu":
            return F.gelu
        else:
            raise ValueError(f"Unrecognized activation function: {act_fun}.")

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
        output_attentions: bool = False,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
            where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query

        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [
                q * torch.sin(weight_index[:, :tgt_len, :] / m),
                q * torch.cos(weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [
                k * torch.sin(weight_index[:, :src_len, :] / m),
                k * torch.cos(weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )

        if self.causal:
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            kv_ = self.attn_dropout(kv_)
            kv_cum = torch.cumsum(kv_, dim=1)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            k_cum = torch.cumsum(k_, dim=1)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            attn_output = qkv / denom.unsqueeze(-1)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )
            attn_weights = None
        else:
            kv_ = torch.einsum("nld,nlm->ndm", k_, v)
            kv_ = self.attn_dropout(kv_)
            z_ = 1 / torch.clamp_min(
                torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
            )
            attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
            )

            # 这一步是新增的，用于计算注意力权重
            if output_attentions:
                attn_weights = torch.einsum("nld,ndm->nlm", q_, kv_)
                attn_weights = torch.softmax(attn_weights, dim=-1)

        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights
        else:
            return attn_output


class CosformerLayer(nn.Module, AbstractTrasnformerLayer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        norm="layernorm",
        norm_first: bool = True,
        causal=False,
    ):
        super().__init__()
        self.self_attn = CosformerAttention(embed_dim=embed_dim, num_heads=num_heads)
        self._ff_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = create_norm(norm, embed_dim)
        self.norm2 = create_norm(norm, embed_dim)
        self.norm_first = norm_first
        self.support_output_attentions = False

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]):
        # x = x.unsqueeze(1)
        x = self.self_attn(x, attn_mask=attn_mask)
        return self.dropout1(x)  # [:, 0, :]

    def forward(self, x_blc, attn_mask=None, output_attentions=False):
        """
        x_blc: BLC

        """
        x = x_blc.transpose(0, 1)  # LBC

        assert (
            output_attentions == False
        ), "output_attentions not implemented for Cosformer"
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        x_out = x.transpose(0, 1)

        return x_out, None
