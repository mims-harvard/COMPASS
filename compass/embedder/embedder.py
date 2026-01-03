# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:31:25 2023

@author: Wanxiang Shen

"""
import torch
import torch.nn as nn
from torch import Tensor
import enum
import math
import os


cwd = os.path.dirname(__file__)


class _GeneInitialization(enum.Enum):

    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> "_GeneInitialization":
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _GeneInitialization]
            raise ValueError(f"initialization must be one of {valid_values}")

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _GeneInitialization.UNIFORM:
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _GeneInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class _GeneExpressionEmbedding(nn.Module):
    """Embeds the gene expression data.

    For one gene, the embeddings consists of two steps:

    * the input abundence features is multiplied by a trainable vector (gene expression embedding)
    * another trainable positinal embedding vector is added (gene postional encoding)

    Note that each gene has its separate pair of trainable embeddings, i.e. the embedding vectors
    are not shared between genes.

    """

    def __init__(
        self,
        n_features: int,
        d_gene: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            n_features: the number of genes
            d_gene: the embed size of one gene
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` means no positional encoding (PE).
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`.

        References:
            *
            https://yura52.github.io/rtdl/stable/_modules/rtdl/modules.html#NumericalFeatureTokenizer

        """
        super().__init__()
        initialization_ = _GeneInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(n_features, d_gene))
        self.bias = nn.Parameter(Tensor(n_features, d_gene)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_gene)

    @property
    def n_genes(self) -> int:
        """The number of genes."""
        return len(self.weight)

    @property
    def d_gene(self) -> int:
        """The size of one gene."""
        return self.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class AbundanceEmbedding(nn.Module):
    def __init__(self, n_genes: int, embed_dim: int, learnable_pe=True) -> None:
        """
        n_genes:  Number of genes
        embed_dim: gene embedding size
        learnable_pe: {True, False}
        """
        super().__init__()
        gee_layer = _GeneExpressionEmbedding(
            n_genes, embed_dim, learnable_pe, "uniform"
        )
        layers = [gee_layer, nn.ReLU()]
        # layers = [gee_layer]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim=876, dim=32, type="gene2vect"):
        """
        type:{'gene2vect', 'umap', 'pumap'}
        """
        super().__init__()
        pefile = os.path.join(cwd, "pe/%s_%s_%s.pt" % (type, input_dim, dim))
        pe = torch.load(pefile)  # 876,32
        pe = pe.unsqueeze(0)  # 1,876,32
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[:, : x.size(1)].requires_grad_(False)
        return pe


class GeneEmbedding(nn.Module):
    def __init__(self, input_dim=876, d_model=32, pos_emb=None):
        """
        pos_emb: {None, 'learnable', 'gene2vect', 'umap'}
        """
        super(GeneEmbedding, self).__init__()

        self.input_dim = input_dim
        self.pos_emb = pos_emb
        self.d_model = d_model

        if pos_emb == "learnable":
            self.abundance_embedder = AbundanceEmbedding(
                input_dim, d_model, learnable_pe=True
            )
            self.positional_encoder = torch.zeros_like
        else:
            self.abundance_embedder = AbundanceEmbedding(
                input_dim, d_model, learnable_pe=False
            )
            if pos_emb is None:
                self.positional_encoder = torch.zeros_like
            else:
                self.positional_encoder = PositionalEncoding(
                    input_dim, d_model, pos_emb
                )

    def forward(self, x):
        """
        Args:
          x: embeddings (batch_size, seq_length)

        Returns:
            gene value embeddings (batch_size, seq_length, d_model) + gene positional embedding (batch_size, seq_length, d_model)
        """
        if (self.pos_emb is None) | (self.pos_emb == "learnable"):
            x = self.abundance_embedder(x)
        else:
            x = self.abundance_embedder(x)  # blc
            pe = self.positional_encoder(x)  # blc
            x = x + pe
        return x
