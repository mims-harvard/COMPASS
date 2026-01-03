import plotly.graph_objects as go
import plotly.io as pio


import torch.nn.functional as F

import os
from tqdm import tqdm
from itertools import chain
import pandas as pd
import numpy as np
import random, torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", font_scale=1.3)
import warnings

warnings.filterwarnings("ignore")


from ..tokenizer import CONCEPT, CONCEPT_palette


CELLTYPE_palette = (
    pd.DataFrame([CONCEPT_palette])
    .T.reset_index()
    .sort_index(ascending=False)
    .set_index("index")[0]
    .to_dict()
)
GENESET_palette = CONCEPT.BroadCelltypePathway.map(CELLTYPE_palette).to_dict()
features = pd.Series(CELLTYPE_palette).index.tolist()


def plot_sankey_diagram(
    model,
    concept2plot=["NKcell"],
    reverse=False,
    scale_imp=False,
    title_text="",
    concept_imp_dict=dict(zip(features, [1 for i in features])),
    topK_vis=10,
    font_size=15,
    width=1200,
    height=600,
    margin=dict(l=10, r=1, t=50, b=10),
    **layout_args
):

    try:
        gene_name = model.feature_name
    except:
        gene_name = model.pretrainer.feature_name

    concept_lineage_map = (
        CONCEPT.drop_duplicates(["BroadCelltypePathway", "Lineage"])
        .set_index("BroadCelltypePathway")
        .Lineage
    )
    LINEA_palette = {
        "Lymphoid_lineage_Bcell": "#80ff00",
        "Lymphoid_lineage_T/NKcell": "#0000ff",
        "Myeloid_lineage": "#ff00ff",
        "Mesenchymal_lineage": "#ffff00",
        "Functional_group": "#ff8000",
    }

    init_gene = "genes"
    init_color = "#0362fc"
    gene_color = "#9e9d93"  #'#eeeee4'
    GENE_palette = dict(zip(gene_name, [gene_color for i in gene_name]))

    # all-in-one-colors
    color_dict = {init_gene: init_color}
    color_dict.update(GENE_palette)
    color_dict.update(GENESET_palette)
    color_dict.update(CELLTYPE_palette)
    color_dict.update(LINEA_palette)

    geneset_weights = (
        model.model.latentprojector.cellpathwayprojector.cellpathway_aggregator.aggregator.attention_weights
    )
    gene_weights = (
        model.model.latentprojector.genesetprojector.geneset_aggregator.aggregator.attention_weights
    )

    concept_name = model.model.latentprojector.cellpathwayprojector.cellpathway_names

    concept_idx = model.model.latentprojector.cellpathwayprojector.cellpathway_indices

    geneset_name = model.model.latentprojector.genesetprojector.genesets_names
    geneset_idx = model.model.latentprojector.genesetprojector.genesets_indices

    all_res = []
    for i, (concept, idx) in enumerate(zip(concept_name, concept_idx)):

        # concept, idx,i
        concept_fi = concept_imp_dict[concept]
        geneset = geneset_name[idx]

        if scale_imp:
            n_genesets = len(geneset)
            n_genes = sum([len(geneset_idx[x]) for x in idx])
            concept_fi = concept_fi * n_genesets  # *n_genes
            # print(concept_fi, n_genesets, n_genes )

        ## geneset->celltype
        geneset_w = geneset_weights["cellpathway_%s" % i].cpu().detach()
        geneset_w = F.softmax(geneset_w, dim=0) * concept_fi
        geneset_w = geneset_w.numpy().reshape(
            -1,
        )
        df2 = pd.DataFrame([geneset, [concept for i in idx], geneset_w]).T
        df2.columns = ["source", "target", "weights"]
        df2["group"] = "geneset->celltype"

        ## gene->geneset
        res = []
        for j, fi in zip(idx, geneset_w):

            target = geneset_name[j]
            sources = gene_name[geneset_idx[j]]

            # print(len(sources))
            if scale_imp:
                fi = fi * len(sources)

            gene_w = gene_weights["geneset_%s" % j].cpu().detach()
            gene_w = F.softmax(gene_w, dim=0)
            gene_w = (
                gene_w.numpy().reshape(
                    -1,
                )
                * fi
            )

            df3 = pd.DataFrame([sources, [target for i in sources], gene_w]).T
            df3.columns = ["source", "target", "weights"]
            df3["group"] = "gene->geneset"
            res.append(df3)
        df3 = pd.concat(res)

        ## concat all
        df = pd.concat(
            [
                df2,
                df3,
            ]
        )  # df1, df4]
        df["concept"] = concept
        df["source_color"] = df["source"].map(color_dict)
        df["target_color"] = df["target"].map(color_dict)
        all_res.append(df)

    dfa = pd.concat(all_res)

    ## start to plot the sankey diagram
    df = dfa[
        dfa.concept.isin(concept2plot)
    ]  # 'Treg','Cytotoxic_Tcell', 'Exhausted_Tcell',

    all_labels = list(set(df.source.unique()) | set(df.target.unique()))
    labels = (
        df.groupby("source").weights.sum().sort_values(ascending=False).index.to_list()
    )
    labels.extend(list(set(all_labels) - set(labels)))

    tail_nodes = (
        df.groupby(["group", "source"])
        .weights.sum()
        .sort_values(ascending=False)
        .loc["gene->geneset"]
        .iloc[topK_vis:]
        .index
    )
    filtered_labels = ["" if label in tail_nodes else label for label in labels]

    colors = pd.Series(labels).map(color_dict).to_list()
    label_num_dict = dict(zip(labels, range(len(labels))))

    # Define the source and target indices
    source = df.source.map(label_num_dict).to_list()
    target = df.target.map(label_num_dict).to_list()
    values = df.weights.to_list()

    df1 = df.groupby(["source", "target"])["weights"].mean().reset_index()

    source = df1.source.map(label_num_dict).to_list()
    target = df1.target.map(label_num_dict).to_list()
    values = df1.weights.to_list()

    if title_text is None:
        title_text = "_".join(concept2plot)

    # Create the Sankey diagram
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.01),
                label=filtered_labels,
                color=colors,
            ),
            link=dict(
                source=target if reverse else source,
                target=source if reverse else target,
                value=values,
            ),
        )
    )

    fig.update_layout(
        title_text=title_text,
        title_font_size=30,
        font_size=font_size,
        width=width,
        height=height,
        margin=margin,
        **layout_args
    )
    return df, fig


def get_projector_weights(model, concept2plot=features):

    try:
        gene_name = model.feature_name
    except:
        gene_name = model.pretrainer.feature_name

    concept_lineage_map = (
        CONCEPT.drop_duplicates(["BroadCelltypePathway", "Lineage"])
        .set_index("BroadCelltypePathway")
        .Lineage
    )
    LINEA_palette = {
        "Lymphoid_lineage_Bcell": "#80ff00",
        "Lymphoid_lineage_T/NKcell": "#0000ff",
        "Myeloid_lineage": "#ff00ff",
        "Mesenchymal_lineage": "#ffff00",
        "Functional_group": "#ff8000",
    }

    init_gene = "genes"
    init_color = "#0362fc"
    gene_color = "#9e9d93"  #'#eeeee4'
    GENE_palette = dict(zip(gene_name, [gene_color for i in gene_name]))

    # all-in-one-colors
    color_dict = {init_gene: init_color}
    color_dict.update(GENE_palette)
    color_dict.update(GENESET_palette)
    color_dict.update(CELLTYPE_palette)
    color_dict.update(LINEA_palette)

    geneset_weights = (
        model.model.latentprojector.cellpathwayprojector.cellpathway_aggregator.aggregator.attention_weights
    )
    gene_weights = (
        model.model.latentprojector.genesetprojector.geneset_aggregator.aggregator.attention_weights
    )

    concept_name = model.model.latentprojector.cellpathwayprojector.cellpathway_names

    concept_idx = model.model.latentprojector.cellpathwayprojector.cellpathway_indices

    geneset_name = model.model.latentprojector.genesetprojector.genesets_names
    geneset_idx = model.model.latentprojector.genesetprojector.genesets_indices

    all_res = []
    for i, (concept, idx) in enumerate(zip(concept_name, concept_idx)):

        # concept, idx,i

        geneset = geneset_name[idx]

        ## geneset->celltype
        geneset_w = geneset_weights["cellpathway_%s" % i].cpu().detach()
        geneset_w = F.softmax(geneset_w, dim=0)
        geneset_w = geneset_w.numpy().reshape(
            -1,
        )
        df2 = pd.DataFrame([geneset, [concept for i in idx], geneset_w]).T
        df2.columns = ["source", "target", "weights"]
        df2["group"] = "geneset->celltype"

        ## gene->geneset
        res = []
        for j, fi in zip(idx, geneset_w):

            target = geneset_name[j]
            sources = gene_name[geneset_idx[j]]

            gene_w = gene_weights["geneset_%s" % j].cpu().detach()
            gene_w = F.softmax(gene_w, dim=0)
            gene_w = gene_w.numpy().reshape(
                -1,
            )

            df3 = pd.DataFrame([sources, [target for i in sources], gene_w]).T
            df3.columns = ["source", "target", "weights"]
            df3["group"] = "gene->geneset"
            res.append(df3)
        df3 = pd.concat(res)

        ## concat all
        df = pd.concat(
            [
                df2,
                df3,
            ]
        )  # df1, df4]
        df["concept"] = concept
        df["source_color"] = df["source"].map(color_dict)
        df["target_color"] = df["target"].map(color_dict)
        all_res.append(df)

    dfa = pd.concat(all_res)

    ## start to plot the sankey diagram
    df = dfa[
        dfa.concept.isin(concept2plot)
    ]  # 'Treg','Cytotoxic_Tcell', 'Exhausted_Tcell',
    return df
