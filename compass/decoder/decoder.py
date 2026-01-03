import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ModuleList, ReLU, Dropout
from copy import deepcopy
import pandas as pd
import numpy as np


import random


def fixseed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class ClassDecoder(nn.Module):
    def __init__(
        self,
        input_dim=44,
        dense_layers=[],
        out_dim=2,
        dropout_p=0.0,
        batch_norms=True,
        seed=42,
    ):
        super().__init__()
        self.seed = seed
        fixseed(seed=seed)

        self.batch_norms = batch_norms  
        self.feature_gate = nn.Parameter(torch.zeros(input_dim))  # logits，
        self.linear = nn.Linear(input_dim, out_dim, bias=True)

        self.use_residual = True
        hidden = dense_layers[-1] if (dense_layers is not None and len(dense_layers) > 0) else input_dim
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden, bias=True),
            nn.Tanh(),                 # use tanh instead of ReLU
            nn.Dropout(dropout_p),
            nn.Linear(hidden, out_dim, bias=True),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0)) 

        # temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, x):
        gate = 2 * torch.sigmoid(self.feature_gate)  # (44,)
        xg = x * gate

        y = self.linear(xg)
        if self.use_residual:
            y = y + torch.tanh(self.res_scale) * self.residual(xg)

        temperature = torch.exp(self.log_temperature)
        return y / temperature


class ProtoNetDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim=2,
        dense_layers=[],
        dropout_p=0.0,      
        batch_norms=True, 
        seed=42,
    ):
        super(ProtoNetDecoder, self).__init__()

        self.seed = seed
        fixseed(seed=seed)

        # ===== layer dims (interface preserved) =====
        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers

        # ===== linear projection (optional, very light) =====
        self.lins = ModuleList()
        for i in range(len(_dense_layers) - 1):
            self.lins.append(
                Linear(_dense_layers[i], _dense_layers[i + 1], bias=False)
            )

        # ===== prototype classifier =====
        last_hidden = _dense_layers[-1]
        self.W = nn.Parameter(torch.Tensor(out_dim, last_hidden))
        self.b = nn.Parameter(torch.zeros(out_dim))

        init.normal_(self.W, mean=0.0, std=0.01)

        # ===== temperature (kept but frozen) =====
        self.log_temperature = nn.Parameter(
            torch.zeros(1), requires_grad=False
        )  # temperature = 1

        self.num_classes = out_dim
        self.feature_dim = input_dim

    def forward(self, x):
        """
        x: [B, input_dim]
        return: logits [B, out_dim]  (NO softmax, PRC-safe)
        """

        # optional light projection (if dense_layers != [])
        for lin in self.lins:
            x = lin(x)  # no activation, no BN, no dropout

        # L2 normalize features & prototypes
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=1)

        # cosine similarity
        logits = torch.matmul(x, W.t()) + self.b

        temperature = torch.exp(self.log_temperature)  # = 1
        return logits / temperature

    def initialize_parameters(self, support_features, support_labels):
        """
        Initialize prototypes with class means (ProtoNet-style).

        support_features: [N, D]
        support_labels:   [N, C] (one-hot)
        """
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = support_labels[:, i] == 1
                if mask.sum() == 0:
                    continue
                class_mean = support_features[mask].mean(dim=0)
                self.W[i].copy_(class_mean)
                


# # Define the Prototypical Network without fine-tuning
class ProtoNetNFTDecoder:

    def __init__(self, temperature=1e-3):

        self.prototype_class_map = {
            "PD": 0,
            "SD": 0,
            "PR": 1,
            "CR": 1,
            1: 1,
            0: 0,
            "R": 1,
            "NR": 0,
        }
        self.temperature = temperature

        
    def fit(self, support_set):
        """
        support_set: the last column is the RECIST label column
        """
        self.label_col = support_set.columns[-1]
        self.feature_col = support_set.columns[:-1]

        unique_recist_labels = support_set[self.label_col].unique()
        out_recist = set(unique_recist_labels) - set(self.prototype_class_map.keys())
        assert len(out_recist) == 0, "Unepxected RECIST labels: %s" % out_recist

        prototype_features = support_set.groupby(self.label_col).median()
        prototype_types = prototype_features.index
        prototype_representation = torch.tensor(prototype_features.values)
        prototype_representation = F.normalize(prototype_representation, p=2, dim=1)
        self.prototype_representation = prototype_representation
        self.prototype_types = prototype_types
        return self

    def transform(self, query_set):
        label_col = query_set.columns[-1]
        assert label_col == self.label_col, "%s is missing!" % self.label_col
    
        X = torch.tensor(query_set[self.feature_col].values, dtype=torch.float)
        X = F.normalize(X, dim=1)
        proto_mean = F.normalize(
            self.prototype_representation.mean(dim=0), dim=0
        )
        proj = (X @ proto_mean.unsqueeze(1)) * proto_mean.unsqueeze(0)
        X = F.normalize(X - proj, dim=1)

        similarities = torch.mm(
            X, self.prototype_representation.T
        )
        similarities = similarities / self.temperature
    
        probabilities = F.softmax(similarities, dim=1)
        probabilities = probabilities.detach().numpy()
    
        dfprob = pd.DataFrame(
            probabilities, index=query_set.index, columns=self.prototype_types
        )
    
        dfprob2 = dfprob.copy()
        dfprob2.columns = dfprob2.columns.map(self.prototype_class_map)
        dfprob2 = dfprob2.T.reset_index().groupby(self.label_col).sum().T
        return dfprob2



class RegDecoder(nn.Module):
    def __init__(
        self,
        input_dim=32,
        dense_layers=[],
        out_dim=1,
        dropout_p=0.0,
        batch_norms=True,
        seed=42,
    ):
        """
        Regression
        """
        super(RegDecoder, self).__init__()
        self.seed = seed
        fixseed(seed=seed)

        ## Input
        self.input_norm = torch.nn.BatchNorm1d(input_dim)

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        self.batch_norms = batch_norms

        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers) - 1):
            lin = Linear(_dense_layers[i], _dense_layers[i + 1])
            self.lins.append(lin)

        ## Batchnorm
        self._batch_norms = ModuleList()
        for i in range(len(_dense_layers) - 1):
            self._batch_norms.append(
                deepcopy(torch.nn.BatchNorm1d(_dense_layers[i + 1]))
            )

        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)

    def forward(self, x):
        if self.batch_norms & (len(self._batch_norms) == 0):
            x = self.input_norm(x)

        for lin, norm in zip(self.lins, self._batch_norms):
            if self.batch_norms:
                x = self.dropout(F.relu(norm(lin(x)), inplace=False))
            else:
                x = self.dropout(F.relu(lin(x), inplace=False))
        y = self.out(x)
        return y
