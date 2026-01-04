import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneSetPoolingAggregator(nn.Module):
    def __init__(self, genesets_indices, pooling_type="mean"):
        """
        Initializes the GeneSetPoolingAggregator module.

        :param genesets_indices: A list of lists, where each inner list contains indices of genes in a gene set.
        :param pooling_type: A string indicating the type of pooling ('mean' or 'max').
        """
        super(GeneSetPoolingAggregator, self).__init__()
        self.genesets_indices = genesets_indices
        self.pooling_type = pooling_type

    def forward(self, gene_output):
        """
        Forward pass of the module. Aggregates gene level features to gene set level features.

        :param gene_output: A tensor of shape (batch_size, num_genes, feature_dim), representing gene-level features.
        :return: A tensor of shape (batch_size, num_genesets, feature_dim), representing aggregated gene set level features.
        """
        batch_size = gene_output.size(0)
        num_genesets = len(self.genesets_indices)
        feature_dim = gene_output.size(2)
        aggregated_features = torch.zeros(
            batch_size, num_genesets, feature_dim, device=gene_output.device
        )

        for i, geneset in enumerate(self.genesets_indices):
            selected_features = gene_output[:, geneset, :]

            if self.pooling_type == "mean":
                aggregated_features[:, i, :] = torch.mean(selected_features, dim=1)
            elif self.pooling_type == "max":
                aggregated_features[:, i, :] = torch.max(selected_features, dim=1)[0]

        return aggregated_features


class GeneSetAttentionAggregator(nn.Module):
    def __init__(self, genesets_indices, softmax_mean=True):
        """
        Initializes the ParametricGeneSetAggregator module.
        :param gene_sets: A list of lists, where each inner list contains indices of genes in a gene set.

        # Example usage
        # Define gene sets
        # Example gene output tensor (batch_size, num_genes, feature_dim)
        >>> gene_output = torch.rand(256, 876, 32)  # Replace with actual gene output tensor
        >>> genesets_indices = [(0, 3, 10), (1, 3, 8), (10, 100, 101, 500)]  # Replace with actual gene sets
        >>> aggregator = ParametricGeneSetAggregator(genesets_indices)
        # Get gene set level features
        >>> gene_set_level_features = aggregator(gene_output)
        >>> gene_set_level_features.shape  # This should be (batch_size, num_gene_sets, feature_dim)
        """
        super(GeneSetAttentionAggregator, self).__init__()

        self.genesets_indices = genesets_indices
        self.softmax_mean = softmax_mean
        # Attention weights for each gene in each gene set
        self.attention_weights = nn.ParameterDict(
            {
                f"geneset_{i}": nn.Parameter(torch.randn(len(gene_set), 1))
                for i, gene_set in enumerate(genesets_indices)
            }
        )

    def forward(self, gene_features):
        """
        Forward pass of the module.

        :param gene_features: A tensor of shape (batch_size, num_genes, feature_dim), representing gene-level features.
        :return: A tensor of shape (batch_size, num_gene_sets, feature_dim), representing gene set level features.
        """
        batch_size = gene_features.size(0)
        gene_set_features = []

        for i, gene_set in enumerate(self.genesets_indices):
            set_features = gene_features[
                :, gene_set, :
            ]  # Extract features for genes in the set

            if self.softmax_mean:
                attention = F.softmax(
                    self.attention_weights[f"geneset_{i}"].expand(batch_size, -1, -1),
                    dim=1,
                )
                weighted_features = set_features * attention
                aggregated_features = torch.sum(weighted_features, dim=1)

            else:
                attention = self.attention_weights[f"geneset_{i}"]
                weighted_features = set_features * attention
                aggregated_features = torch.mean(weighted_features, dim=1)

            gene_set_features.append(aggregated_features)

        gene_set_features = torch.stack(gene_set_features, dim=1)

        return gene_set_features


class GeneSetAggregator(nn.Module):
    def __init__(self, genesets_indices, mode="attention", pooling_type="mean"):
        """
        Initializes the CellPathwayAggregator module.
        :param genesets_indices: A list of lists, each sublist contains indices of genes for a specific gene set.
        :param mode: The aggregation mode ('pooling' or 'attention').
        :param pooling_type: The type of pooling ('mean' or 'max'), used if mode is 'pooling'.

        Example
        ===============
        >>> gene_output = torch.rand(10, 876, 32)  # Replace with actual gene output tensor
        >>> genesets_indices = [(0, 3, 10), (1, 3, 8), (10, 100, 101, 500)]  # Replace with actual gene sets
        >>> aggregator = GeneSetAggregator(genesets_indices, mode='attention')
        >>> aggregator(gene_output).shape

        """
        super(GeneSetAggregator, self).__init__()
        self.mode = mode
        self.genesets_indices = genesets_indices

        if mode == "pooling":
            self.aggregator = GeneSetPoolingAggregator(genesets_indices, pooling_type)
        elif mode == "attention":
            self.aggregator = GeneSetAttentionAggregator(genesets_indices)
        else:
            raise ValueError("Invalid mode. Use 'pooling' or 'attention'.")

    def forward(self, gene_features):
        """
        Forward pass of the module.
        :param gene_features: A tensor of shape (batch_size, num_genes, feature_dim).
        :return: A tensor of shape (batch_size, num_cellpathways).
        """
        geneset_feats = self.aggregator(gene_features)
        return geneset_feats
