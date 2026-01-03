# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:31:25 2023

@author: Wanxiang Shen

"""

import torch.nn as nn
from torch.nn.modules.container import ModuleList
from torchvision.ops import MLP

import copy, torch
from .layer import CosformerLayer, PerformerLayer, FlowformerLayer
from .layer import VanillaTransformerLayer, FlashTransformerLayer
from ..embedder import GeneEmbedding, _GeneInitialization

from .layer.norm import create_norm


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):

    def __init__(
        self,
        encoder_type="transformer",
        d_model=32,
        dim_feedforward=64,
        nhead=2,
        num_layers=1,
        dropout=0,
        **kwargs,
    ):

        super(Encoder, self).__init__()

        self.encoder_type = encoder_type
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_head = d_model // nhead  # 16
        self.dropout = dropout

        if encoder_type == "cosformer":
            encoder_layer = CosformerLayer(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, **kwargs
            )
        elif encoder_type == "performer":
            encoder_layer = PerformerLayer(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, **kwargs
            )
        elif encoder_type == "Vanillatransformer":
            encoder_layer = VanillaTransformerLayer(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, **kwargs
            )
        elif encoder_type == "flowformer":
            encoder_layer = FlowformerLayer(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, **kwargs
            )
        elif encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                **kwargs,
            )

        elif encoder_type == "flashformer":
            encoder_layer = FlashTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                **kwargs,
            )

        else:
            raise NotImplementedError(
                f"Not implemented transformer type: {encoder_type}"
            )

        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, x, output_attentions=False):

        att_list = []
        for l in range(self.num_layers):
            if self.encoder_type == "transformer":
                x = self.layers[l](x)
                att = None
            else:
                x, att = self.layers[l](x, output_attentions=output_attentions)
            att_list.append(att)

        return x, att_list


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_cancer_types=33,
        encoder_type="transformer",
        input_dim=15672,
        nhead=2,
        d_model=32,
        num_layers=2,
        dropout=0.0,
        dim_feedforward=32,
        pos_emb="learnable",
        **kwargs,
    ):
        """
        encoder_type: {'transformer', 'reformer', 'performer'}
        pos_emb: {None, 'learnable', 'gene2vect', 'umap'}
        d_model: {16,32,64,128,512}
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.num_cancer_types = num_cancer_types
        self.input_dim = input_dim
        self.nhead = nhead
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pos_emb = pos_emb

        self.pid_token_embedder = nn.Parameter(torch.randn(1, d_model))
        initialization_ = _GeneInitialization.from_str("uniform")
        initialization_.apply(self.pid_token_embedder, d_model)

        self.cancer_token_embedder = nn.Embedding(num_cancer_types, d_model)
        self.gene_token_embedder = GeneEmbedding(input_dim, d_model, pos_emb)

        self.encoder = Encoder(
            encoder_type=encoder_type,
            d_model=d_model,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
            **kwargs,
        )

    def forward(self, x, output_attentions=False):

        cancer_types = x[:, 0].long()  # first column is the cancer type
        genes = x[:, 1:]  #

        # expand patient/sample/dataset token (equals to the CLS token)
        pid_embed = self.pid_token_embedder.expand(x.size(0), -1).unsqueeze(1)  # B,1, C
        # convert cancer types
        cancer_embed = self.cancer_token_embedder(cancer_types).unsqueeze(1)  # B,1, C
        # gene embeddings
        gene_embed = self.gene_token_embedder(genes)  # B, L, C

        # concat dataset_embed,cancer_embed,
        transformer_input = torch.cat(
            [pid_embed, cancer_embed, gene_embed], dim=1
        )  # B, L+2, C

        x, attn = self.encoder(transformer_input, output_attentions=output_attentions)

        if output_attentions:
            return x, attn
        else:
            return x


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, dense_layers=[512, 256, 128], dropout=0.2):
        super().__init__()
        self.encoder = MLP(
            in_channels=input_dim,
            hidden_channels=dense_layers,
            norm_layer=torch.nn.BatchNorm1d,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
