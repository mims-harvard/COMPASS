"""
Personal COMPASS Response Map (CRMap) utilities
==============================================

This module provides end-to-end helpers to construct and render a patient-level
**Personal COMPASS Response Map (CRMap)** — a layered graph that links
expression (gene TPM) → gene score → geneset score → concept (celltype) score → output (NR/R).

Functions
---------
- `prepare_crmap_data`:
    Run COMPASS inference and assemble all data required to build a CRMap.
    Specifically, this helper:
      1) loads the trained COMPASS model and runs prediction on an input
         expression matrix (`dfcx`);
      2) extracts latent scores at the gene, geneset, and concept (celltype)
         layers (`dfgn`, `dfgs`, `dfct`);
      3) (optionally) z-scores the above matrices over patients to standardize
         scale;
      4) derives correlation-based edge weights for each CRMap layer:
         genetpm→gene, gene→geneset, geneset→celltype, celltype→output;
      5) returns the four edge tables together with the tensors needed
         downstream.  The return order is exactly what
         `personal_crmap_data` expects.

- `personal_crmap_data`:
    Build a *patient-specific*, long-form, layered edge list that attaches
    per-node values for a chosen patient. This filters to requested high-level
    concepts (celltypes) and optionally keeps only the top-K genes per geneset.
    The output is a single DataFrame concatenating edges from all CRMap layers
    with `source_value` / `target_value` columns filled for the patient.

- `draw_personal_crmap`:
    Render the CRMap for a single patient as a left-to-right layered graph with
    curved edges. Node color encodes node value (with separate colormaps for
    genetpm vs. other layers); edge color encodes layer-specific weights.
    Multiple layout/label parameters control spacing, curvature, thresholds,
    and styling.

Typical workflow
----------------
1. Call `prepare_crmap_data(dfcx, ...)` to obtain:
   `(celltype2output, geneset2celltype, gene2geneset, genetpm2gene,
     dfgn, dfgs, dfct, dfpred, dfcx)`.
2. Call `personal_crmap_data(patient_id=..., concept2plot=..., TopK_gene=..., 
   celltype2output=..., geneset2celltype=..., gene2geneset=..., genetpm2gene=..., 
   dfgn=..., dfgs=..., dfct=..., dfpred=..., dfcx=...)` to build the
   patient-specific layered table.
3. Call `draw_personal_crmap(df, concept2plot=..., ...)` to visualize.

Inputs & assumptions
--------------------
- `dfcx` (patients × genes): input expression (e.g., TPM; columns are gene symbols,
  index are patient IDs).
- A trained COMPASS model loadable via `loadcompass(...)`, and projector/encoder
  weights accessible via `get_projector_weights(model)`.
- Latent extractions `model.extract(..., with_gene_level=True)` must return
  `(dfgn, dfgs, dfct)` aligned to the same patients as `dfcx`.
- Predictions `model.predict(...)` must yield a two-column `dfpred` that can be
  interpreted as `['NR','R']` (the helper normalizes column names if necessary).

Example
-------
>>> # 1) Prepare CRMap data from expression
>>> (celltype2output, geneset2celltype, gene2geneset, genetpm2gene,
...  dfgn, dfgs, dfct, dfpred, dfcx) = prepare_crmap_data(dfcx, z_scale=True)
>>>
>>> # 2) Build patient-specific layered table
>>> crmap_df = personal_crmap_data(
...     patient_id="P001",
...     concept2plot=["IFNg_pathway", "Cytotoxic_Tcell", 'Endothelial', 'TGFb_pathway'],
...     TopK_gene=3,
...     celltype2output=celltype2output,
...     geneset2celltype=geneset2celltype,
...     gene2geneset=gene2geneset,
...     genetpm2gene=genetpm2gene,
...     dfgn=dfgn, dfgs=dfgs, dfct=dfct, dfpred=dfpred, dfcx=dfcx
... )
>>>
>>> # 3) Draw the CRMap
>>> fig = draw_personal_crmap(crmap_df, concept2plot=["NKcell", "Reference"])
>>> fig.savefig("P001_CRMap.svg", bbox_inches="tight", dpi=150)

Dependencies
------------
- pandas, numpy, scikit-learn (for z-scoring, pairwise correlations),
- matplotlib (required), seaborn (optional for global style),
- project-specific utilities: `loadcompass`, `get_projector_weights`.

Notes
-----
- `prepare_crmap_data` returns values in the exact order expected by
  `personal_crmap_data`:
  `(celltype2output, geneset2celltype, gene2geneset, genetpm2gene,
    dfgn, dfgs, dfct, dfpred, dfcx)`.
- For dense graphs, consider increasing thresholds in
  `draw_personal_crmap(..., label_threshold=..., genetpm_gene_edge_threshold=...)`
  or reducing `TopK_gene` in `personal_crmap_data`.
"""

import warnings
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from .sankey import get_projector_weights
from .loader import loadcompass


def prepare_crmap_data(
    dfcx: pd.DataFrame,
    model_path_name: str = "./finetuner_pft_all.pt",
    map_location: str = "cpu",
    z_scale: bool = True,
    concept_palette: Optional[Dict[str, str]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run COMPASS inference and construct CRMap edge tables.

    Parameters
    ----------
    dfcx : DataFrame (patients x genes)
        Input expression (e.g., TPM; columns = gene symbols; index = patient IDs).
    model_path_name : str, default "finetuner_pft_all.pt"
        Path to the fine-tuned COMPASS checkpoint.
    map_location : str, default "cpu"
        Torch map_location (e.g., "cpu", "cuda:0").
    z_scale : bool, default True
        Z-score each of dfcx/dfgn/dfgs/dfct (column-wise over patients) before
        computing correlations and return the scaled versions.
    concept_palette : dict[str,str], optional
        Optional color mapping for concepts (celltypes) in celltype2output (key: concept name).

    Returns
    -------
    celltype2output : DataFrame
    geneset2celltype : DataFrame
    gene2geneset : DataFrame
    genetpm2gene : DataFrame
    dfgn : DataFrame
    dfgs : DataFrame
    dfct : DataFrame
    dfpred : DataFrame
    dfcx : DataFrame

    Notes
    -----
    - The four returned edge tables follow CRMap layers:
      genetpm→gene, gene→geneset, geneset→celltype, celltype→output.
    - `dfpred` keeps the model's original two columns (often ['$P_{NR}$','$P_{R}$']).
      Internally, we temporarily map them to ['NR','R'] to compute celltype→output correlations.
    """
    # -------- 1) Inference --------
    model = loadcompass(model_path_name, map_location=map_location)
    _, dfpred = model.predict(dfcx, batch_size=128)
    dfgn, dfgs, dfct = model.extract(dfcx, batch_size=128, with_gene_level=True)

    # Keep original dfpred for return; internally map to ['NR','R'] for correlation
    dfpred_out = dfpred.copy()
    dfp = dfpred_out.copy()
    if set(dfp.columns) == {"$P_{NR}$", "$P_{R}$"}:
        dfp.columns = ["NR", "R"]
    elif set(dfp.columns) != {"NR", "R"}:
        # minimal fallback: take first two columns as NR/R
        dfp = dfp.iloc[:, :2].copy()
        dfp.columns = ["NR", "R"]

    dfw = get_projector_weights(model).copy()
    dfw = dfw[["source", "target", "weights", "group", "concept"]]

    # -------- 2) Optional z-scoring (column-wise) --------
    def _z(df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler().set_output(transform="pandas")
        return scaler.fit_transform(df)

    if z_scale:
        dfcx_z = _z(dfcx)
        dfgn_z = _z(dfgn)
        dfgs_z = _z(dfgs)
        dfct_z = _z(dfct)
        dfp_z = _z(dfp)  # only used for celltype→output correlation
    else:
        dfcx_z, dfgn_z, dfgs_z, dfct_z, dfp_z = dfcx, dfgn, dfgs, dfct, dfp

    # -------- 3) Align patients across all tables --------
    common_idx = dfct_z.index
    for m in (dfgs_z, dfgn_z, dfp_z, dfcx_z):
        common_idx = common_idx.intersection(m.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping patients across dfct/dfgs/dfgn/dfpred/dfcx.")

    dfct_ = dfct_z.loc[common_idx]
    dfgs_ = dfgs_z.loc[common_idx]
    dfgn_ = dfgn_z.loc[common_idx]
    dfcx_ = dfcx_z.loc[common_idx]
    dfp_ = dfp_z.loc[common_idx]

    # -------- 4) Build CRMap edge tables --------
    # celltype -> output
    corr_ct_out = 1 - pairwise_distances(dfct_.T, dfp_.T, metric="correlation")
    celltype2output = (
        pd.DataFrame(corr_ct_out, index=dfct_.columns, columns=["NR", "R"])
        .sort_values("R", ascending=False)
        .unstack()
        .reset_index()
    )
    celltype2output.columns = ["target", "source", "weights"]  # target in {'NR','R'}
    celltype2output["group"] = "celltype->output"
    celltype2output["concept"] = celltype2output["source"]
    celltype2output["source_color"] = (
        celltype2output["source"].map(concept_palette)
        if concept_palette is not None
        else "#9e9d93"
    )
    celltype2output["target_color"] = "#9e9d93"

    # geneset -> celltype
    geneset2celltype = dfw[dfw.group == "geneset->celltype"][
        ["source", "target", "group", "concept"]
    ].copy()
    corr_gs_ct = 1 - pairwise_distances(dfgs_.T, dfct_.T, metric="correlation")
    tmp = (
        pd.DataFrame(np.abs(corr_gs_ct), index=dfgs_.columns, columns=dfct_.columns)
        .T.unstack()
        .loc[geneset2celltype.set_index(["source", "target"]).index]
        .reset_index()
    )
    tmp.columns = ["source", "target", "weights"]
    geneset2celltype["weights"] = tmp["weights"].values

    # gene -> geneset
    gene2geneset = dfw[dfw.group == "gene->geneset"][
        ["source", "target", "group", "concept"]
    ].copy()
    corr_gn_gs = 1 - pairwise_distances(dfgn_.T, dfgs_.T, metric="correlation")
    tmp = (
        pd.DataFrame(np.abs(corr_gn_gs), index=dfgn_.columns, columns=dfgs_.columns)
        .T.unstack()
        .loc[gene2geneset.set_index(["source", "target"]).index]
        .reset_index()
    )
    tmp.columns = ["source", "target", "weights"]
    gene2geneset["weights"] = tmp["weights"].values

    # genetpm -> gene
    used_genes = [
        g
        for g in gene2geneset["source"].unique()
        if g in dfcx_.columns and g in dfgn_.columns
    ]
    if len(used_genes) == 0:
        raise ValueError(
            "No overlap between genes in dfcx/dfgn and 'gene->geneset' edges."
        )
    X = dfcx_[used_genes].values  # TPM
    Y = dfgn_[used_genes].values  # gene scores
    corr_cx_gn = 1 - pairwise_distances(X.T, Y.T, metric="correlation")
    attn = np.abs(corr_cx_gn)
    genetpm2gene = (
        pd.DataFrame(attn, columns=used_genes, index=used_genes).unstack().reset_index()
    )
    genetpm2gene.columns = ["source", "target", "weights"]
    genetpm2gene["source"] = genetpm2gene["source"] + "@TPM"
    genetpm2gene["group"] = "genetpm->gene"
    genetpm2gene["concept"] = "Gene"
    genetpm2gene["source_color"] = "#9e9d93"
    genetpm2gene["target_color"] = "#9e9d93"

    # -------- 5) Return in the exact order personal_crmap_data expects --------
    # (celltype2output, geneset2celltype, gene2geneset, genetpm2gene, dfgn, dfgs, dfct, dfpred, dfcx)
    return (
        celltype2output,
        geneset2celltype,
        gene2geneset,
        genetpm2gene,
        dfgn_,
        dfgs_,
        dfct_,
        dfpred_out,
        dfcx_,
    )


def personal_crmap_data(
    patient_id,
    concept2plot=["NKcell", "Reference"],
    TopK_gene=1,
    celltype2output=None,
    geneset2celltype=None,
    gene2geneset=None,
    genetpm2gene=None,
    dfgn=None,
    dfgs=None,
    dfct=None,
    dfpred=None,
    dfcx=None,
):
    """
    Build the patient-specific layered edge table used by `draw_personal_crmap`.

    Given a patient identifier and the projector/encoder tables that link layers
    (genetpm→gene→geneset→celltype→output), this function assembles a long-form
    DataFrame of edges with associated node values for the chosen patient. It
    also restricts the visualization to a subset of concepts (celltypes) and,
    optionally, the top-k contributing genes per geneset.

    Parameters
    ----------
    patient_id : hashable
        Identifier that exists as an index in all node-value frames (`dfgn`,
        `dfgs`, `dfct`, `dfpred`, `dfcx`).
    concept2plot : list of str, default ['NKcell', 'Reference']
        Celltype (high-level concepts) to include in the final graph.
    TopK_gene : int, default 1
        For each geneset, keep the top-k genes by absolute `weights` in the
        `gene2geneset` table.
    celltype2output : pandas.DataFrame
        Edge list mapping celltype → output. Required columns:
        ['source', 'target', 'weights']; `source` is a celltype; `target` in {'R', 'NR'}.
    geneset2celltype : pandas.DataFrame
        Edge list mapping geneset → celltype. Required columns:
        ['source', 'target', 'weights'].
    gene2geneset : pandas.DataFrame
        Edge list mapping gene → geneset. Required columns:
        ['source', 'target', 'weights'].
    genetpm2gene : pandas.DataFrame
        Edge list mapping (gene@TPM) → gene. Required columns:
        ['source', 'target', 'weights'] where `source` uses the convention
        '<gene>@TPM'.
    dfgn : pandas.DataFrame
        Gene-level node values (e.g., z-scored gene scores). Index are patient
        IDs; columns are gene symbols.
    dfgs : pandas.DataFrame
        Geneset-level node values. Index are patient IDs; columns are genesets.
    dfct : pandas.DataFrame
        Celltype (concept)–level node values. Index are patient IDs; columns are
        celltypes in your ontology.
    dfpred : pandas.DataFrame
        Output probabilities/scores. Index are patient IDs; columns must contain
        two entries that will be assigned to ['NR', 'R'] for visualization.
    dfcx : pandas.DataFrame
        Gene TPM–level node values (input). Index are patient IDs; columns are
        gene symbols (not including '@TPM'); values typically z-scored TPM.

    Returns
    -------
    pandas.DataFrame
        Long-form edge table concatenating four groups:
        - 'celltype->output'
        - 'geneset->celltype'
        - 'gene->geneset'
        - 'genetpm->gene'
        With the following columns:
        ['source', 'target', 'weights', 'group', 'source_value', 'target_value'].

    Raises
    ------
    ValueError
        If any required frame is None, if `patient_id` is not found, or if
        required columns are missing.

    Notes
    -----
    - `dfpred.loc[patient]` is reindexed to ['NR', 'R'] to standardize output ordering.
    - Genes are filtered to `TopK_gene` per geneset by descending 'weights'.
    - For `genetpm2gene`, `source` is matched after stripping '@TPM' to align
      with `dfcx` (gene TPM values).
    """
    # ---- original function body of personal_camap_data goes here unchanged ----
    # Simply paste your existing implementation body here.
    # Only the function name & docstring changed.
    if any(
        v is None
        for v in [
            celltype2output,
            geneset2celltype,
            gene2geneset,
            genetpm2gene,
            dfgn,
            dfgs,
            dfct,
            dfpred,
            dfcx,
        ]
    ):
        raise ValueError("Please provide all required data.")
    try:
        _ = dfcx.loc[patient_id]
        patient = patient_id
    except KeyError:
        raise ValueError(f"Patient ID {patient_id} not found in clinical data.")

    dfp1 = celltype2output[celltype2output["source"].isin(concept2plot)].reset_index(
        drop=True
    )
    dfp1["source_value"] = dfp1.source.map(dfct.loc[patient])
    pred = dfpred.loc[patient]
    pred.index = ["NR", "R"]
    dfp1["target_value"] = dfp1.target.map(pred)

    dfp2 = geneset2celltype[geneset2celltype.target.isin(concept2plot)].reset_index(
        drop=True
    )
    dfp2["source_value"] = dfp2.source.map(dfgs.loc[patient])
    dfp2["target_value"] = dfp2.target.map(dfct.loc[patient])

    dfp3 = gene2geneset[gene2geneset.target.isin(dfp2.source)]
    dfp3 = (
        dfp3.groupby("target")
        .apply(lambda x: x.sort_values("weights", ascending=False).iloc[:TopK_gene])
        .reset_index(drop=True)
    )
    dfp3["source_value"] = dfp3.source.map(dfgn.loc[patient])
    dfp3["target_value"] = dfp3.target.map(dfgs.loc[patient])

    dfp4 = genetpm2gene[
        genetpm2gene.source.apply(lambda x: x.split("@")[0]).isin(dfp3.source.unique())
    ]
    dfp4 = dfp4[dfp4.target.isin(dfp3.source.unique())]
    dfp4["source_value"] = dfp4.source.apply(lambda x: x.split("@")[0]).map(
        dfcx.loc[patient]
    )
    dfp4["target_value"] = dfp4.target.map(dfgn.loc[patient])

    data_r = pd.concat([dfp1, dfp2, dfp3, dfp4]).reset_index(drop=True)
    return data_r


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import get_cmap, ScalarMappable
import numpy as np
import seaborn as sns
from itertools import chain
import io


def draw_personal_crmap(
    df,
    concept2plot,
    figsize=(16, 15),
    fontsize=15,
    layer_node_sizes=[0.5, 0.5, 1, 3, 5],
    max_rad=0.25,
    show_geneset_name=False,
    layer_node_gaps={"celltype": 0.15, "geneset": 0.1, "output": 0.2},
    layer_spacing=[1.2, 1, 1, 0.8],
    node_label_alignments={
        "genetpm": -0.01,
        "gene": -0.01,
        "geneset": 0.01,
        "celltype": 0.02,
        "output": 0.02,
    },
    layer_edge_styles={"celltype->output": {"linestyle": "--", "arrowstyle": "-"}},
    genetpm_node_color_map={"cmap": None, "vmin": -2, "vmax": 2},
    compass_node_color_map={"cmap": None, "vmin": 0, "vmax": 1},
    label_threshold={
        "genetpm": 1,
        "gene": 0.5,
        "geneset": 0.5,
        "celltype": -10,
        "output": -10,
    },
    label_color_same_as_node=True,
    genetpm2gene_edge_colormap={"cmap": None, "vmin": 0.6, "vmax": 1.0},
    projector_edge_colormap={"cmap": None, "vmin": 0.0, "vmax": 0.5},
    edge_line_width=[0.2, 0.3, 0.3, 0.5],
    genetpm_gene_edge_threshold=0.6,
    edge_activate={"threshold": 10, "color": "red"},
):
    """
    Render a multi-layer, curved-edge Personal COMPASS Response Map (CRMap).

    This function draws a left-to-right layered graph with nodes arranged by
    layer in the order:
        genetpm → gene → geneset → celltype → output
    Node colors reflect node values (two independent colormaps for genetpm vs
    the rest), and edges are colored by their layer-specific colormap. Several
    layout and styling knobs are exposed to control spacing, label density, and
    curvature.

    Parameters
    ----------
    df : pandas.DataFrame
        The patient-specific layered edge table returned by `personal_crmap_data`.
        Required columns:
        - 'source', 'target', 'weights'
        - 'group' with values in {'genetpm->gene','gene->geneset','geneset->celltype','celltype->output'}
        - 'source_value', 'target_value'
    concept2plot : list of str
        High-level concepts (celltypes) to prioritize in ordering (right side).
        If present, they are placed in reverse order for vertical alignment.
    figsize : tuple, default (16, 15)
        Matplotlib figure size (inches).
    fontsize : int, default 15
        Base font size for labels and colorbar titles.
    layer_node_sizes : list of float, default [0.5, 0.5, 1, 3, 5]
        Relative node sizes per layer (genetpm→…→output). Values are scaled by 100 internally.
    max_rad : float, default 0.25
        Maximum edge curvature (arc radius) in `FancyArrowPatch` connection styles.
    show_geneset_name : bool, default False
        Whether to draw geneset labels (can be dense).
    layer_node_gaps : dict, optional
        Vertical spacing per layer (fraction of axis height between neighboring nodes).
    layer_spacing : list of float, default [1.2, 1, 1, 0.8]
        Horizontal spacing between consecutive layers (linearly normalized to [0,1]).
    node_label_alignments : dict, optional
        Per-layer horizontal text offsets; negative values left-align, positive
        right-align relative to node center.
    layer_edge_styles : dict, optional
        Mapping from '<source_layer->target_layer>' to style dicts with keys
        'linestyle' and 'arrowstyle'.
    genetpm_node_color_map : dict, optional
        Node colormap spec for genetpm layer: {'cmap': name or None, 'vmin': float, 'vmax': float}.
        If `cmap` is None, a default blue–white–red map is used.
    compass_node_color_map : dict, optional
        Node colormap spec for gene/geneset/celltype/output layers. If `cmap`
        is None, a default white→yellow→orange→red map is used.
    label_threshold : dict, optional
        Minimum node value to show a text label per layer. Set very low (e.g.,
        -10) to always show; increase to sparsify labels.
    label_color_same_as_node : bool, default True
        If True, gene/TPM labels adopt the node color, improving saliency.
    genetpm2gene_edge_colormap : dict, optional
        Colormap spec for genetpm→gene edges.
    projector_edge_colormap : dict, optional
        Colormap spec for projector edges (gene→geneset→celltype→output).
    edge_line_width : list of float, default [0.2, 0.3, 0.3, 0.5]
        Line widths per edge group following the layer order above.
    genetpm_gene_edge_threshold : float, default 0.6
        Prunes genetpm→gene edges with |weight| below this threshold (except
        the self-named “gene@TPM → gene” identity edges).
    edge_activate : dict, optional
        If provided as {'threshold': float, 'color': str}, re-colors edges with
        weight > threshold to the given color.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure (axes are created internally).

    Notes
    -----
    - Layer order is inferred from `df['group']` and fixed to
      genetpm→gene→geneset→celltype→output.
    - The function builds node positions with per-layer vertical spacing controls
      (`layer_node_gaps`) and per-layer horizontal spacing (`layer_spacing`).
    - Four colorbars are added (genetpm nodes, compass nodes, encoder edges, projector edges).
    - To save the figure, call `fig.savefig(...)` on the returned `Figure`.

    Examples
    --------
    >>> camap = personal_camap_data(
    ...     patient_id="P001",
    ...     concept2plot=["CD8_T", "NKcell"],
    ...     TopK_gene=3,
    ...     celltype2output=celltype2output,
    ...     geneset2celltype=geneset2celltype,
    ...     gene2geneset=gene2geneset,
    ...     genetpm2gene=genetpm2gene,
    ...     dfgn=dfgn, dfgs=dfgs, dfct=dfct, dfpred=dfpred, dfcx=dfcx,
    ... )
    >>> fig = draw_personal_camap(camap, concept2plot=["CD8_T", "NKcell"], figsize=(14, 12))
    >>> fig.savefig("P001_camap.svg", bbox_inches="tight", dpi=150)

    """
    # ---- original function body of draw_personal_camap goes here unchanged ----
    # Paste your existing implementation body here; only the name/docstring changed.
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    layer_links_order = [
        "genetpm->gene",
        "gene->geneset",
        "geneset->celltype",
        "celltype->output",
    ]
    df["weights_abs"] = df["weights"].abs()
    layer_node_order_series = df.groupby(["group", "target"]).apply(
        lambda x: x.sort_values("weights", ascending=True).source.tolist()
    )
    layer_node_order = {}
    try:
        layer_node_order["celltype"] = concept2plot[
            ::-1
        ]  # layer_node_order_series.loc[('celltype->output', 'R')]
    except KeyError:
        layer_node_order["celltype"] = []
    try:
        geneset_order_series = (
            df[df.group == "geneset->celltype"]
            .groupby("target")
            .apply(lambda x: x.sort_values("weights").source.tolist())
            .loc[layer_node_order["celltype"]]
        )
        geneset_order = list(chain.from_iterable(geneset_order_series))
        layer_node_order["geneset"] = geneset_order
    except KeyError:
        layer_node_order["geneset"] = []
    layer_node_order["output"] = ["NR", "R"]

    if "group" not in df.columns:
        raise ValueError("DataFrame must contain a 'group' column.")
    geneset_celltype_mapping = (
        df[df.group == "geneset->celltype"].set_index("source")["target"].to_dict()
    )

    if genetpm_node_color_map.get("cmap") is None:
        genetpm_colors = ["blue", "white", "red"]
        cmap_genetpm = LinearSegmentedColormap.from_list(
            "genetpm_default", genetpm_colors
        )
    else:
        cmap_genetpm = plt.get_cmap(genetpm_node_color_map["cmap"])
    vmin_genetpm = genetpm_node_color_map.get("vmin", None)
    vmax_genetpm = genetpm_node_color_map.get("vmax", None)

    if compass_node_color_map.get("cmap") is None:
        compass_colors = ["white", "yellow", "orange", "red"]
        cmap_compass = LinearSegmentedColormap.from_list(
            "compass_default", compass_colors
        )
    else:
        cmap_compass = plt.get_cmap(compass_node_color_map["cmap"])
    vmin_compass = compass_node_color_map.get("vmin", None)
    vmax_compass = compass_node_color_map.get("vmax", None)

    if genetpm2gene_edge_colormap.get("cmap") is None:
        g2g_colors = [
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#4292c6",
            "#2171b5",
            "#08519c",
            "#08306b",
            "#041f47",
        ]
        g2g_cmap = LinearSegmentedColormap.from_list(
            "genetpm2gene_edge_default", g2g_colors
        )
    else:
        g2g_cmap = plt.get_cmap(genetpm2gene_edge_colormap["cmap"])
    g2g_vmin = genetpm2gene_edge_colormap["vmin"]
    g2g_vmax = genetpm2gene_edge_colormap["vmax"]
    g2g_norm = Normalize(vmin=g2g_vmin, vmax=g2g_vmax)

    if projector_edge_colormap.get("cmap") is None:
        proj_colors = [
            "#C7E9B4",
            "#7FCDBB",
            "#41B6C4",
            "#1D91C0",
            "#225EA8",
            "#253494",
            "#081D58",
            "#08306b",
            "#041F47",
        ]
        proj_cmap = LinearSegmentedColormap.from_list(
            "projector_edge_default", proj_colors
        )
    else:
        proj_cmap = plt.get_cmap(projector_edge_colormap["cmap"])
    proj_vmin = projector_edge_colormap["vmin"]
    proj_vmax = projector_edge_colormap["vmax"]
    proj_norm = Normalize(vmin=proj_vmin, vmax=proj_vmax)

    df[["source_layer", "target_layer"]] = df["group"].str.split("->", expand=True)

    all_layers = []
    for link in layer_links_order:
        layers = link.split("->")
        for layer in layers:
            if layer not in all_layers:
                all_layers.append(layer)
    n_layers = len(all_layers)

    if "source_value" not in df.columns or "target_value" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'source_value' and 'target_value' columns."
        )
    node_values = {}
    node_layers = {}
    for idx, rowd in df.iterrows():
        node_values[rowd["source"]] = rowd["source_value"]
        node_values[rowd["target"]] = rowd["target_value"]
        node_layers[rowd["source"]] = rowd["source_layer"]
        node_layers[rowd["target"]] = rowd["target_layer"]

    layer_nodes_dict = {}
    for node, layer in node_layers.items():
        layer_nodes_dict.setdefault(layer, set()).add(node)

    layer_node_sizes = [size * 100 for size in layer_node_sizes]

    for layer in all_layers:
        nodes = layer_nodes_dict.get(layer, set())
        if layer in layer_node_order:
            ordered_nodes = [node for node in layer_node_order[layer] if node in nodes]
            unordered_nodes = [node for node in nodes if node not in ordered_nodes]
            layer_nodes = ordered_nodes + unordered_nodes
        else:
            if layer == "geneset" and geneset_celltype_mapping:
                geneset_groups = {}
                for node in nodes:
                    celltype = geneset_celltype_mapping.get(node, "Unknown")
                    geneset_groups.setdefault(celltype, []).append(node)
                layer_nodes = []
                for celltype in sorted(geneset_groups.keys()):
                    group_nodes = sorted(geneset_groups[celltype], reverse=True)
                    layer_nodes.extend(group_nodes)
            else:
                layer_nodes = sorted(nodes, reverse=True)
        layer_nodes_dict[layer] = layer_nodes

    genetpm_layer = "genetpm"
    genetpm_nodes = layer_nodes_dict.get(genetpm_layer, [])
    compass_nodes = []
    for layer in all_layers:
        if layer != genetpm_layer:
            compass_nodes.extend(layer_nodes_dict.get(layer, []))

    genetpm_values = [node_values[n] for n in genetpm_nodes] if genetpm_nodes else []
    compass_values = [node_values[n] for n in compass_nodes] if compass_nodes else []

    if vmin_genetpm is None and genetpm_values:
        vmin_genetpm = min(genetpm_values)
    if vmax_genetpm is None and genetpm_values:
        vmax_genetpm = max(genetpm_values)
    if vmin_compass is None and compass_values:
        vmin_compass = min(compass_values)
    if vmax_compass is None and compass_values:
        vmax_compass = max(compass_values)

    norm_genetpm = Normalize(vmin=vmin_genetpm, vmax=vmax_genetpm)
    norm_compass = Normalize(vmin=vmin_compass, vmax=vmax_compass)

    # Use layer_spacing to determine layer positions
    if len(layer_spacing) < n_layers - 1:
        layer_spacing = list(layer_spacing) + [1] * (n_layers - 1 - len(layer_spacing))
    layer_spacing = layer_spacing[: n_layers - 1]

    x_positions = [0]
    for sp in layer_spacing:
        x_positions.append(x_positions[-1] + sp)
    max_pos = x_positions[-1] if x_positions else 1
    if max_pos == 0:
        max_pos = 1
    x_positions = [x / max_pos for x in x_positions]

    layer_positions = {layer: x_positions[idx] for idx, layer in enumerate(all_layers)}

    node_positions = {}
    for idx, layer_name in enumerate(all_layers):
        nodes = layer_nodes_dict.get(layer_name, [])
        n_nodes = len(nodes)
        x_position = layer_positions[layer_name]
        node_size = layer_node_sizes[idx]

        if n_nodes > 1:
            if layer_name in layer_node_gaps:
                gap_y = layer_node_gaps[layer_name]
                total_height = gap_y * (n_nodes - 1)
                if total_height > 1:
                    gap_y = 1.0 / (n_nodes - 1)
                y_start = 0.5 - (gap_y * (n_nodes - 1) / 2)
                y_positions = [y_start + i * gap_y for i in range(n_nodes)]
            else:
                y_positions = np.linspace(0, 1, n_nodes)
        else:
            y_positions = [0.5]

        y_positions = [min(max(y, 0), 1) for y in y_positions]
        positions = list(zip([x_position] * n_nodes, y_positions))

        if layer_name == "genetpm":
            c_map = cmap_genetpm
            c_norm = norm_genetpm
        else:
            c_map = cmap_compass
            c_norm = norm_compass

        x_positions_plot = [pos[0] for pos in positions]
        y_positions_plot = [pos[1] for pos in positions]
        node_colors = [c_map(c_norm(node_values[n])) for n in nodes]
        # node_colors = [mcolors.to_hex(c_map(c_norm(node_values[n]))) for n in nodes]

        ax.scatter(
            x_positions_plot,
            y_positions_plot,
            s=node_size,
            zorder=3,
            linewidths=0.5,
            edgecolors="k",
            facecolors=node_colors,
        )
        for n, (xx, yy) in zip(nodes, positions):
            node_positions[n] = (xx, yy)

        if label_threshold and layer_name in label_threshold:
            threshold = label_threshold[layer_name]
            offset = node_label_alignments.get(layer_name, -0.02)
            i = 0
            for n, (xx, yy) in zip(nodes, positions):
                if layer_name == "geneset" and not show_geneset_name:
                    continue

                if layer_name == "genetpm":
                    node_value = abs(node_values[n])
                else:
                    node_value = node_values[n]

                if node_value > threshold:
                    if offset < 0:
                        text_x = xx + offset
                        ha = "right"
                    elif offset > 0:
                        text_x = xx + offset
                        ha = "left"
                    else:
                        text_x = xx
                        ha = "center"
                    label_text = str(n).replace("@TPM", "")

                    if layer_name == "output":
                        label_text = "$P_{%s} = %s$" % (
                            label_text,
                            round(node_value, 2),
                        )

                    if label_color_same_as_node and (
                        layer_name == "genetpm"
                    ):  # or layer_name == 'gene'
                        ax.text(
                            text_x,
                            yy,
                            label_text,
                            fontsize=fontsize,
                            ha=ha,
                            va="center",
                            color=node_colors[i],
                        )
                    else:
                        ax.text(
                            text_x,
                            yy,
                            label_text,
                            fontsize=fontsize,
                            ha=ha,
                            va="center",
                        )
                i += 1

        layer_name_map = {
            "genetpm": "INPUT" + "\n$X_{GeneTPM}$",
            "gene": "Gene" + "\nscore: $S_{Gene}$",
            "geneset": "Granular concept" + "\nscore: $S_{Geneset}$",
            "celltype": "High-level concept" + "\nscore: $S_{Concept}$",
            "output": "OUTPUT" + "\n$P_{(R|NR)}$",
        }

        if layer_name != "output":
            ax.text(
                x_position,
                1.07,
                layer_name_map.get(layer_name),
                fontsize=fontsize + 2,
                ha="center",
                va="center",
            )
        else:
            ax.text(
                x_position,
                0.5,
                layer_name_map.get(layer_name),
                fontsize=fontsize + 2,
                ha="center",
                va="center",
            )

    for idx, rowd in df.iterrows():
        source = rowd["source"]
        target = rowd["target"]
        weight = rowd["weights"]
        source_layer = rowd["source_layer"]
        target_layer = rowd["target_layer"]
        layer_connection = f"{source_layer}->{target_layer}"

        # threshold pruning for genetpm->gene
        if (
            layer_connection == "genetpm->gene"
            and abs(weight) < genetpm_gene_edge_threshold
            and (str(source).replace("@TPM", "") != target)
        ):
            continue

        if layer_connection == "genetpm->gene":
            edge_norm_value = g2g_norm(weight)
            current_edge_color = g2g_cmap(edge_norm_value)
            lw = edge_line_width[layer_links_order.index(layer_connection)]
        else:
            edge_norm_value = proj_norm(weight)
            current_edge_color = proj_cmap(edge_norm_value)
            lw = edge_line_width[layer_links_order.index(layer_connection)]

        if edge_activate:
            threshold = edge_activate.get("threshold", None)
            override_color = edge_activate.get("color", None)
            if threshold is not None and override_color is not None:
                if weight > threshold:
                    current_edge_color = override_color

        (x_a, y_a) = node_positions[source]
        (x_b, y_b) = node_positions[target]
        delta_y = y_b - y_a
        rad = max_rad * delta_y
        rad = max(-max_rad, min(max_rad, rad))

        edge_style = layer_edge_styles.get(
            layer_connection, {"linestyle": "-", "arrowstyle": "-"}
        )
        line_style = edge_style.get("linestyle", "-")
        arrow_style = edge_style.get("arrowstyle", "-")

        con = FancyArrowPatch(
            (x_a, y_a),
            (x_b, y_b),
            arrowstyle=arrow_style,
            linestyle=line_style,
            connectionstyle=f"arc3,rad={rad}",
            color=current_edge_color,
            linewidth=lw,
            mutation_scale=10,
        )
        ax.add_patch(con)

    fig.subplots_adjust(right=0.88)

    sm_genetpm = ScalarMappable(cmap=cmap_genetpm, norm=norm_genetpm)
    sm_genetpm.set_array([])
    cax_genetpm = fig.add_axes([0.94, 0.74, 0.015, 0.1])
    cbar_genetpm = plt.colorbar(sm_genetpm, cax=cax_genetpm, orientation="vertical")
    cbar_genetpm.set_label(
        "Gene TPM\nZ-score", rotation=90, labelpad=5, fontsize=fontsize
    )
    cbar_genetpm.ax.tick_params(labelsize=fontsize)
    cbar_genetpm.set_ticks([vmin_genetpm, vmax_genetpm])
    cbar_genetpm.set_ticklabels([f"{vmin_genetpm:.1f}", f"{vmax_genetpm:.1f}"])

    sm_compass = ScalarMappable(cmap=cmap_compass, norm=norm_compass)
    sm_compass.set_array([])
    cax_compass = fig.add_axes([0.94, 0.56, 0.015, 0.1])
    cbar_compass = plt.colorbar(sm_compass, cax=cax_compass, orientation="vertical")
    cbar_compass.set_label(
        "COMPASS score\nZ-score", rotation=90, labelpad=5, fontsize=fontsize
    )
    cbar_compass.ax.tick_params(labelsize=fontsize)
    cbar_compass.set_ticks([vmin_compass, vmax_compass])
    cbar_compass.set_ticklabels([f"{vmin_compass:.1f}", f"{vmax_compass:.1f}"])

    sm_g2g = ScalarMappable(cmap=g2g_cmap, norm=g2g_norm)
    sm_g2g.set_array([])
    cax_g2g = fig.add_axes([0.94, 0.37, 0.015, 0.1])
    cbar_g2g = plt.colorbar(sm_g2g, cax=cax_g2g, orientation="vertical")
    cbar_g2g.set_label(
        "Encoder\nedge weights", rotation=90, labelpad=5, fontsize=fontsize
    )
    cbar_g2g.ax.tick_params(labelsize=fontsize)
    cbar_g2g.set_ticks([g2g_vmin, g2g_vmax])
    cbar_g2g.set_ticklabels([f"{g2g_vmin:.1f}", f"{g2g_vmax:.1f}"])

    sm_proj = ScalarMappable(cmap=proj_cmap, norm=proj_norm)
    sm_proj.set_array([])
    cax_proj = fig.add_axes([0.94, 0.19, 0.015, 0.1])
    cbar_proj = plt.colorbar(sm_proj, cax=cax_proj, orientation="vertical")
    cbar_proj.set_label(
        "Projector\nedge weights", rotation=90, labelpad=5, fontsize=fontsize
    )
    cbar_proj.ax.tick_params(labelsize=fontsize)
    cbar_proj.set_ticks([proj_vmin, proj_vmax])
    cbar_proj.set_ticklabels([f"{proj_vmin:.1f}", f"{proj_vmax:.1f}"])

    plt.close(fig)
    return fig


if __name__ == "__main__":
    # Example
    # crmap_data = personal_crmap_data(
    #     patient_id=patient_id,
    #     concept2plot=concept2plot,
    #     TopK_gene=TopK_gene,
    #     celltype2output=celltype2output,
    #     geneset2celltype=geneset2celltype,
    #     gene2geneset=gene2geneset,
    #     genetpm2gene=genetpm2gene,
    #     dfgn=dfgn, dfgs=dfgs, dfct=dfct, dfpred=dfpred, dfcx=dfcx,
    # )
    # crmap_fig = draw_personal_crmap(crmap_data, concept2plot=concept2plot)
    # crmap_fig.savefig("CRMap_example.svg", bbox_inches="tight", dpi=150)
    pass
