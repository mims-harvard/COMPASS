from .cellpathwayaggregator import CellPathwayAggregator
from .genesetaggregator import GeneSetAggregator
from .genesetscorer import GeneSetScorer
from ..tokenizer import CONCEPT, TOKENS_SHORT, TOKENS_LONG


import torch, math
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np


cwd = os.path.dirname(__file__)


class GeneSetProjector(nn.Module):
    def __init__(
        self,
        GENESET,
        geneset_feature_dim,
        geneset_agg_mode="attention",
        geneset_score_mode="linear",
    ):
        super(GeneSetProjector, self).__init__()

        self.geneset_feature_dim = geneset_feature_dim
        self.geneset_agg_mode = geneset_agg_mode
        self.geneset_score_mode = geneset_score_mode

        self.GENESET = GENESET
        self.genesets_indices = GENESET.tolist()
        self.genesets_names = GENESET.index

        self.geneset_aggregator = GeneSetAggregator(
            self.genesets_indices, self.geneset_agg_mode
        )
        self.geneset_scorer = GeneSetScorer(
            self.geneset_feature_dim, self.geneset_score_mode
        )

        # self.norm = nn.LayerNorm(len(self.genesets_indices))
        # nn.BatchNorm1d(n)

    def forward(self, x):
        geneset_feats = self.geneset_aggregator(x)

        geneset_scores = self.geneset_scorer(geneset_feats)
        # geneset_scores = F.normalize(geneset_scores, p=2., dim=1)
        return geneset_scores


class CellPathwayProjector(nn.Module):

    def __init__(self, CELLPATHWAY, cellpathway_agg_mode="pooling"):
        super(CellPathwayProjector, self).__init__()
        self.cellpathway_agg_mode = cellpathway_agg_mode
        self.CELLPATHWAY = CELLPATHWAY
        self.cellpathway_indices = CELLPATHWAY.tolist()
        self.cellpathway_names = CELLPATHWAY.index
        self.cellpathway_aggregator = CellPathwayAggregator(
            self.cellpathway_indices, mode=self.cellpathway_agg_mode
        )

    def forward(self, x):
        cellpathway_scores = self.cellpathway_aggregator(x)
        return cellpathway_scores


class DisentangledProjector(nn.Module):

    def __init__(
        self,
        gene_num,
        gene_feature_dim,
        proj_pid=True,
        proj_cancer_type=True,
        geneset_agg_mode="attention",
        geneset_score_mode="linear",
        cellpathway_agg_mode="attention",
    ):

        super(DisentangledProjector, self).__init__()

        TOKENS = TOKENS_LONG if len(TOKENS_LONG) == gene_num + 2 else TOKENS_SHORT
        TOKEN_IDX_MAP = (
            pd.Series(TOKENS).reset_index().set_index(0)["index"].astype(int).to_dict()
        )

        def _genes_to_idxs(genes):
            TOKEN_IDX_MAP
            gene_idxs = []
            for i in genes.split(":"):
                idx = TOKEN_IDX_MAP.get(i)
                if idx != None:
                    gene_idxs.append(idx)
            return gene_idxs

        GENESET = CONCEPT.Genes.apply(_genes_to_idxs)
        CELLPATHWAY = (
            CONCEPT.reset_index()
            .groupby("BroadCelltypePathway")
            .apply(lambda x: x.index.tolist())
        )

        ##sort the concept by the order
        GENESET = GENESET.loc[CONCEPT.sort_values("GeneSet_index").index]
        c_order = (
            CONCEPT[["BroadCelltypePathway", "Concept_index"]]
            .drop_duplicates()
            .sort_values("Concept_index")["BroadCelltypePathway"]
            .tolist()
        )
        CELLPATHWAY = CELLPATHWAY.loc[c_order]

        assert "Reference" in CELLPATHWAY.index, "Reference must in CELLPATHWAY!"
        # Move 'Reference' to the end
        noref = CELLPATHWAY[CELLPATHWAY.index != "Reference"]
        ref = CELLPATHWAY[CELLPATHWAY.index == "Reference"]
        CELLPATHWAY = noref._append(ref)

        self.GENESET = GENESET
        self.CELLPATHWAY = CELLPATHWAY

        geneset_name_list = GENESET.index.tolist()
        celltype_name_list = CELLPATHWAY.index.tolist()

        self.proj_pid = proj_pid
        self.proj_cancer_type = proj_cancer_type

        self.gene_num = gene_num
        self.gene_feature_dim = gene_feature_dim
        self.geneset_agg_mode = geneset_agg_mode
        self.geneset_score_mode = geneset_score_mode
        self.cellpathway_agg_mode = cellpathway_agg_mode
        self.genesetprojector = GeneSetProjector(
            GENESET,
            self.gene_feature_dim,
            self.geneset_agg_mode,
            self.geneset_score_mode,
        )
        self.cellpathwayprojector = CellPathwayProjector(
            CELLPATHWAY, self.cellpathway_agg_mode
        )

        self.patientprojector = GeneSetScorer(self.gene_feature_dim, "linear")
        self.cancerprojector = GeneSetScorer(self.gene_feature_dim, "linear")

        if proj_pid and proj_cancer_type:
            PROJCOLS = ["PID", "CANCER"]

        elif proj_pid and not proj_cancer_type:
            PROJCOLS = ["PID"]

        elif not proj_pid and proj_cancer_type:
            PROJCOLS = ["CANCER"]

        else:
            PROJCOLS = []

        geneset_proj_cols = PROJCOLS.copy()
        geneset_proj_cols.extend(geneset_name_list)

        cellpathway_proj_cols = PROJCOLS.copy()
        cellpathway_proj_cols.extend(celltype_name_list)

        self.cellpathway_proj_cols = cellpathway_proj_cols
        self.geneset_proj_cols = geneset_proj_cols

        ## reference genes
        ref_gene_ids = []
        for refset in GENESET.iloc[CELLPATHWAY["Reference"]].to_list():
            ref_gene_ids.extend(refset)

        ## reference sets
        ref_geneset_ids = []
        for rgs in GENESET.iloc[CELLPATHWAY["Reference"]].index:
            idx = geneset_proj_cols.index(rgs)
            ref_geneset_ids.append(idx)

        ## reference cellpathways
        ref_celltype_ids = [cellpathway_proj_cols.index("Reference")]

        self.ref_gene_ids = ref_gene_ids
        self.ref_celltype_ids = ref_celltype_ids
        self.ref_geneset_ids = ref_geneset_ids

    def forward(self, x):

        # x size is  B, L+2, 1
        pid_encoding = x[:, 0:1, :]  # take the learnbale patient id token
        cancer_encoding = x[:, 1:2, :]  # take the cancer_type token
        gene_encoding = x[:, 2:, :]  # take the gene encoding

        geneset_scores = self.genesetprojector(gene_encoding)
        cellpathway_scores = self.cellpathwayprojector(geneset_scores)

        if self.proj_pid and self.proj_cancer_type:
            cancer_scores = self.cancerprojector(cancer_encoding)
            pid_scores = self.patientprojector(pid_encoding)
            geneset_proj = torch.cat([pid_scores, cancer_scores, geneset_scores], dim=1)
            cellpathway_proj = torch.cat(
                [pid_scores, cancer_scores, cellpathway_scores], dim=1
            )

        elif self.proj_pid and not self.proj_cancer_type:
            pid_scores = self.patientprojector(pid_encoding)
            geneset_proj = torch.cat([pid_scores, geneset_scores], dim=1)
            cellpathway_proj = torch.cat([pid_scores, cellpathway_scores], dim=1)

        elif not self.proj_pid and self.proj_cancer_type:
            cancer_scores = self.cancerprojector(cancer_encoding)
            geneset_proj = torch.cat([cancer_scores, geneset_scores], dim=1)
            cellpathway_proj = torch.cat([cancer_scores, cellpathway_scores], dim=1)

        else:
            geneset_proj = geneset_scores
            cellpathway_proj = cellpathway_scores

        return geneset_proj, cellpathway_proj


class GeneSetPlaceholderAggregator(nn.Module):
    def __init__(self, genes_num, genesets_num):
        """
        Initializes the GeneSetPlaceholderAggregator module.
        :param gene_sets: A list of lists, where each inner list contains indices of genes in a gene set.

        """
        super(GeneSetPlaceholderAggregator, self).__init__()

        self.genesets_num = genesets_num
        self.genes_num = genes_num

        # Attention weights for each gene in each placeholder gene set
        self.attention_weights = nn.ParameterDict(
            {
                f"Placeholder_geneset_{i}": nn.Parameter(torch.randn(genes_num, 1))
                for i, gene_set in enumerate(range(genesets_num))
            }
        )

    def forward(self, x):
        transformed = self.transformer(x)
        gene_set_outputs = []
        for i in range(self.genesets_num):
            weight = self.attention_weights[f"Placeholder_geneset_{i}"]
            gene_set_output = transformed * weight
            gene_set_outputs.append(gene_set_output.sum(dim=1))

        return torch.stack(gene_set_outputs, dim=1)

    def adaptive_prune(self):
        with torch.no_grad():
            all_weights = torch.cat(
                [param.view(-1) for param in self.attention_weights.values()]
            )
            threshold = torch.quantile(torch.abs(all_weights), 0.90)  # 保留最重要的10%
            for name, param in self.attention_weights.items():
                mask = torch.abs(param) >= threshold
                param.mul_(mask)


## need to be revised to fit the cls and cancer type token
class EntangledProjector(nn.Module):
    def __init__(self, gene_feature_dim, mode="mean"):
        """
        reduce: {'mean', 'max', 'cls'}
        """
        super(EntangledProjector, self).__init__()
        self.mode = mode
        self.gene_feature_dim = gene_feature_dim

    def forward(self, x):
        if self.mode == "mean":
            x = torch.mean(x, dim=-1)
        elif self.mode == "max":
            x = torch.max(x, dim=-1)[0]
        else:
            ValueError("Invalid pooling type. Use 'mean' or 'max'.")
        return x
