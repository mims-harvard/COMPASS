import torch
import torch.nn as nn
import torch.nn.functional as F


class CellPathwayPoolingAggregator(nn.Module):
    def __init__(self, cellpathway_indices, pooling_type="mean"):
        """
        Initializes the CellPathwayAggregatorPooling module.
        :param cellpathway_indices: A list of lists, each sublist contains indices of gene sets for a specific cell pathway.
        :param pooling_type: A string indicating the type of pooling ('mean' or 'max').
        """
        super(CellPathwayPoolingAggregator, self).__init__()
        self.cellpathway_indices = cellpathway_indices
        self.pooling_type = pooling_type

    def forward(self, gene_set_features):
        """
        Forward pass of the module.
        :param gene_set_features: A tensor of shape (batch_size, num_gene_sets, feature_dim).
        :return: A tensor of shape (batch_size, num_cellpathways).
        """
        device = gene_set_features.device
        batch_size = gene_set_features.size(0)
        num_cellpathways = len(self.cellpathway_indices)
        aggregated_features = torch.zeros(batch_size, num_cellpathways, device=device)

        for i, indices in enumerate(self.cellpathway_indices):
            if self.pooling_type == "mean":
                aggregated_features[:, i] = torch.mean(
                    gene_set_features[:, indices], dim=1
                )
            elif self.pooling_type == "max":
                aggregated_features[:, i] = torch.max(
                    gene_set_features[:, indices], dim=1
                )[0]

        return aggregated_features


class CellPathwayAttentionAggregator(nn.Module):
    def __init__(self, cellpathway_indices, softmax_mean=True):
        """
        Initializes the CellPathwayAttentionAggregator module.

        :param cellpathway_indices: A list of lists, each list contains indices of cell types/pathways in a cellpathway set.
        """
        super(CellPathwayAttentionAggregator, self).__init__()
        self.cellpathway_indices = cellpathway_indices
        self.softmax_mean = softmax_mean

        # Attention weights for each cell type/pathway in each cellpathway set
        self.attention_weights = nn.ParameterDict(
            {
                f"cellpathway_{i}": nn.Parameter(torch.randn(len(cellpathway), 1))
                for i, cellpathway in enumerate(cellpathway_indices)
            }
        )

    def forward(self, geneset_features):
        """
        Forward pass of the module. Aggregates cell type/pathway level features to cellpathway set level features using attention mechanism.
        :param geneset_features:
        :return: A tensor of shape (batch_size, num_cellpathway_sets), representing aggregated cellpathway set level features.
        """
        batch_size = geneset_features.size(0)
        num_cellpathway_sets = len(self.cellpathway_indices)
        aggregated_features = torch.zeros(
            batch_size, num_cellpathway_sets, device=geneset_features.device
        )

        for i, cellpathway in enumerate(self.cellpathway_indices):
            set_features = geneset_features[:, cellpathway]
            if self.softmax_mean:
                attention_scores = F.softmax(
                    self.attention_weights[f"cellpathway_{i}"], dim=0
                )
                weighted_features = set_features * attention_scores.T
                aggregated_features[:, i] = torch.sum(weighted_features, dim=1)
            else:
                attention_scores = self.attention_weights[f"cellpathway_{i}"]
                weighted_features = set_features * attention_scores.T
                aggregated_features[:, i] = torch.mean(weighted_features, dim=1)
        return aggregated_features


class CellPathwayAggregator(nn.Module):
    def __init__(self, cellpathway_indices, mode="pooling", pooling_type="mean"):
        """
        Initializes the CellPathwayAggregator module.
        :param cellpathway_indices: A list of lists, each sublist contains indices of gene sets for a specific cell pathway.
        :param mode: The aggregation mode ('pooling' or 'attention').
        :param pooling_type: The type of pooling ('mean' or 'max'), used if mode is 'pooling'.
        """
        super(CellPathwayAggregator, self).__init__()
        self.mode = mode

        if mode == "pooling":
            self.aggregator = CellPathwayPoolingAggregator(
                cellpathway_indices, pooling_type
            )
        elif mode == "attention":
            self.aggregator = CellPathwayAttentionAggregator(cellpathway_indices)
        else:
            raise ValueError("Invalid mode. Use 'pooling' or 'attention'.")

    def forward(self, geneset_features):
        """
        Forward pass of the module.
        :param gene_set_features: A tensor of shape (batch_size, num_gene_sets).
        :return: A tensor of shape (batch_size, num_cellpathways).
        """
        # print(geneset_features.device)

        cellpathway_features = self.aggregator(geneset_features)

        # print(cellpathway_features.device)
        return cellpathway_features
