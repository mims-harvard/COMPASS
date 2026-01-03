from .scorer import score, score2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns

sns.set(style="white", font_scale=1.5)


def plot_embed_with_label(
    dfp,
    label_col=["cancer_type"],
    label_type=["c"],
    orders=[None],
    figsize=(10, 10),
    metric="correlation",
    spread=1,
    n_neighbors=5,
    min_dist=0.5,
    s=5,
    random_state=123,
    verbose=False,
    return_coord=False,
    cmap="bright",
    **kwargs,
):
    """
    dfp: dataframe of samples x genes, with a column contains batch information
    label_col: list of labels to be used
    label_type: list of the label types, 'c' for categorical label, 'r' for continous label
    """

    glist = dfp.columns[~dfp.columns.isin(label_col)]
    if len(glist) == 2:
        df2d = dfp[glist]
    else:
        mp = UMAP(
            spread=spread,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )  # , metric='correlation'
        embed = mp.fit_transform(dfp[glist])
        df2d = pd.DataFrame(embed, columns=["UMAP1", "UMAP2"], index=dfp.index)
    col1, col2 = df2d.columns
    df2d = df2d.join(dfp[label_col])

    figs = []
    for label, t, order in zip(label_col, label_type, orders):
        fig, ax = plt.subplots(figsize=figsize)
        if t == "c":
            cohort_order = order  # df2d.groupby(label).size().sort_values().index
            if cohort_order is None:
                cohort_order = df2d.groupby(label).size().sort_values().index
            colors = sns.color_palette(cmap, len(cohort_order)).as_hex()

            for bt, c in zip(cohort_order, colors):
                dfp1 = df2d[df2d[label] == bt]
                ax.scatter(dfp1[col1], dfp1[col2], label=bt, s=s, c=c)
                # print(color)
            if len(cohort_order) <= 16:
                ax.legend(
                    loc="center left",
                    ncol=1,
                    prop={"size": 12},
                    bbox_to_anchor=(1, 0.5),
                )  #
            else:
                ax.legend(
                    loc="center left",
                    ncol=3,
                    prop={"size": 12},
                    bbox_to_anchor=(1, 0.5),
                )  #
        else:
            ax.scatter(
                df2d[col1], df2d[col2], label=label, s=s, c=df2d[label], cmap=cmap
            )
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax.tick_params(
            bottom="on",
            left="off",
            labelleft="on",
            labelbottom="on",
            pad=-0.6,
        )
        sns.despine(top=True, right=True, left=False, bottom=False)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(label)

        figs.append(fig)

    if not return_coord:
        return figs

    else:
        return figs, df2d


def plot_performance(y_true, y_prob, y_pred):

    from sklearn.metrics import confusion_matrix

    roc, prc, f1, acc, mcc = score2(y_true, y_prob, y_pred)
    dfp = pd.DataFrame([y_true, y_prob, y_pred]).T
    dfp.columns = ["Label", "Pred. Prob.", "Pred_label"]
    dfp.Label = dfp.Label.map({0: "NR", 1: "R"})
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp_and_fn = cf_matrix.sum(1)
    tp_and_fp = cf_matrix.sum(0)
    tp = cf_matrix.diagonal()
    precision = tp / tp_and_fp
    recall = tp / tp_and_fn
    precision, recall = precision[0], recall[0]

    palette = sns.color_palette("rainbow", 12)

    colors = palette.as_hex()
    boxpalette = {"NR": colors[1], "R": colors[-3]}
    swarmpalette = {"NR": colors[2], "R": colors[-3]}

    fig, axes = plt.subplots(
        ncols=3,
        nrows=1,
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [4, 1, 3]},
        sharex=False,
        sharey=False,
    )

    ax2, ax1, ax3 = axes

    ###################################
    order = ["R", "NR"]
    sns.boxplot(
        data=dfp,
        x="Label",
        y="Pred. Prob.",
        fliersize=0.0,
        width=0.5,
        order=order,
        ax=ax1,
        palette=boxpalette,
        saturation=0.8,
        boxprops={"facecolor": "None"},
    )
    sns.stripplot(
        data=dfp,
        x="Label",
        y="Pred. Prob.",
        ax=ax1,
        size=3,
        order=order,
        palette=boxpalette,
        edgecolor="k",
        linewidth=0.1,
    )

    ax1.xaxis.tick_bottom()  # x axis on top
    ax1.xaxis.set_label_position("bottom")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", labelrotation=60)

    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.yaxis.tick_left()  # x axis on top
    ax1.spines[["right", "top"]].set_visible(False)

    ###################################
    group_names = ["True R", "False NR", "False R", "True NR"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    labels = [
        f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)
    ]  # ,group_percentages
    labels = np.asarray(labels).reshape(2, 2)

    cf_df = pd.DataFrame(cf_matrix, index=["R", "NR"], columns=["R", "NR"])
    sns.heatmap(cf_df, annot=labels, fmt="", cmap="Blues", ax=ax2, cbar=False)

    ax2.xaxis.tick_bottom()  # x axis on top
    ax2.xaxis.set_label_position("bottom")
    # ax2.set_xlabel("Predicted Label")
    ax2.tick_params(axis="x", labelrotation=60)

    ax2.yaxis.tick_left()  # x axis on top
    ax2.yaxis.set_label_position("left")
    ax2.set_ylabel("True Label")
    ax2.tick_params(axis="y", labelrotation=60)

    ###################################
    dfpp = pd.DataFrame(
        [mcc, roc, prc, f1, acc], index=["MCC", "ROC", "PRC", "F1", "ACC"]
    )

    dfpp.plot(kind="barh", ax=ax3, legend=False, color="b", alpha=0.5)
    ax3.yaxis.tick_left()  # x axis on top
    ax3.yaxis.set_label_position("left")
    ax3.set_xticks([0.0, 0.5, 1.0])
    ax3.xaxis.tick_bottom()  # x axis on top
    ax3.xaxis.set_label_position("bottom")

    for y, x in dfpp[0].reset_index(drop=True).items():
        ax3.text(x, y - 0.15, "%.2f" % x)
    ax3.spines[["right", "top"]].set_visible(False)

    fig.tight_layout(pad=1.5)
    return fig
