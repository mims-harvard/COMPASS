import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingAggregator(nn.Module):
    def __init__(self, agg_indices_series, pooling_type="mean"):
        """
        Initializes the PoolingAggregator module.
        :param agg_indices_series: A pd.Series, each_index is the name the aggregated features, each value is the list of the node to be aggreagated
        :param pooling_type: A string indicating the type of pooling ('mean' or 'max').
        """
        super(PoolingAggregator, self).__init__()
        self.agg_indices_series = agg_indices_series
        self.agg_indices = agg_indices_series.tolist()
        self.pooling_type = pooling_type

    def forward(self, gene_set_features):
        """
        Forward pass of the module.
        :param gene_set_features: A tensor of shape (batch_size, num_gene_sets).
        :return: A tensor of shape (batch_size, num_agg_features).

        """
        device = gene_set_features.device
        batch_size = gene_set_features.size(0)
        num_agg_features = len(self.agg_indices)
        aggregated_features = torch.zeros(batch_size, num_agg_features, device=device)

        for i, indices in enumerate(self.agg_indices):
            if self.pooling_type == "mean":
                aggregated_features[:, i] = torch.mean(
                    gene_set_features[:, indices], dim=1
                )
            elif self.pooling_type == "max":
                aggregated_features[:, i] = torch.max(
                    gene_set_features[:, indices], dim=1
                )[0]

        return aggregated_features


class AttentionAggregator(nn.Module):
    def __init__(self, agg_indices_series):
        """
        Initializes the AttentionAggregator module.

        :param agg_indices_series: A pd.Series, each_index is the name the aggregated features, each value is the list of the node to be aggreagated
        """
        super(AttentionAggregator, self).__init__()

        self.agg_indices_series = agg_indices_series
        self.agg_indices = agg_indices_series.tolist()

        # Attention weights for each cell type/pathway in each cellpathway set
        self.attention_weights = nn.ParameterDict(
            {
                f"{name}": nn.Parameter(torch.randn(len(getset_idx), 1))
                for name, getset_idx in agg_indices_series.items()
            }
        )

    def forward(self, gene_set_features):
        """
        Forward pass of the module. Aggregates cell type/pathway level features to cellpathway set level features using attention mechanism.
        :param gene_set_features: A tensor of shape (batch_size, num_gene_sets).
        :return: A tensor of shape (batch_size, num_cellpathway_sets), representing aggregated cellpathway set level features.
        """
        batch_size = gene_set_features.size(0)
        num_agg_features = len(self.agg_indices)
        aggregated_features = torch.zeros(
            batch_size, num_agg_features, device=gene_set_features.device
        )
        i = 0
        for name, getset_idx in self.agg_indices_series.items():
            set_features = gene_set_features[:, getset_idx]
            attention_scores = F.softmax(self.attention_weights[f"{name}"], dim=0)
            weighted_features = set_features * attention_scores.T
            aggregated_features[:, i] = torch.sum(weighted_features, dim=1)
            i += 1

        return aggregated_features
