import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneSetScorePooling(nn.Module):
    def __init__(self, pooling_type="mean"):
        """
        Initializes the GeneSetScorePooling module for gene set score calculation.
        :param pooling_type: A string indicating the type of pooling ('mean' or 'max').
        """
        super(GeneSetScorePooling, self).__init__()
        self.pooling_type = pooling_type

    def forward(self, x):
        """
        Forward pass of the module.
        :param x: A tensor of shape (batch_size, num_gene_sets, feature_dim).
        :return: A tensor of shape (batch_size, num_gene_sets).
        """
        if self.pooling_type == "mean":
            return torch.mean(x, dim=2)
        elif self.pooling_type == "max":
            return torch.max(x, dim=2)[0]
        else:
            raise ValueError("Invalid pooling type. Use 'mean' or 'max'.")


class DRNet(nn.Module):
    def __init__(self, feature_dim):
        super(DRNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  #
        x = self.fc2(x)
        return x


class GeneSetScoreLinear(nn.Module):
    def __init__(self, feature_dim):
        """
        Initializes the GeneSetScoreLinear module for gene set score calculation.
        :param feature_dim: The dimension of features for each gene set.
        """
        super(GeneSetScoreLinear, self).__init__()
        self.fc = nn.Linear(
            feature_dim, 1
        )  # Linear layer to transform features to a single score
        # self.fc = DRNet(feature_dim)

    def forward(self, x):
        """
        Forward pass of the module.
        :param x: A tensor of shape (batch_size, num_gene_sets, feature_dim).
        :return: A tensor of shape (batch_size, num_gene_sets).
        """
        batch_size, num_gene_sets, _ = x.shape
        x = x.view(-1, x.size(-1))  # Flatten the last two dimensions for linear layer
        scores = self.fc(x)
        # scores = F.relu(scores) # score greater than zero
        scores = scores.view(
            batch_size, num_gene_sets
        )  # Reshape to original batch and gene set dimensions
        return scores


class GeneSetScorer(nn.Module):
    def __init__(self, feature_dim, mode="linear", pooling_type="mean"):
        """
        Initializes the GeneSetScore module for gene set score calculation.
        :param mode: A string indicating the type of pooling ('mean' or 'max', or 'linear').

        # Example usage
        ==================
        >>> gene_set_level_features = torch.rand(10, 3, 32)  # Assume gene_set_level_features has shape
        >>> scorer = GeneSetScorer(32, mode='linear')
        >>> scores = scorer(gene_set_level_features)  # shape will be (256, 3)
        >>> scores.shape
        """

        super(GeneSetScorer, self).__init__()
        self.mode = mode
        self.feature_dim = feature_dim

        if mode == "pooling":
            self.genesetscorer = GeneSetScorePooling(pooling_type)
        elif mode == "linear":
            self.genesetscorer = GeneSetScoreLinear(feature_dim)
        else:
            raise ValueError("Invalid mode type. Use 'mean','max' or 'linear'. ")

    def forward(self, x):
        scores = self.genesetscorer(x)
        return scores
