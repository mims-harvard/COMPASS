# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""
import torch
import torch.utils.data as Torchdata
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas(ascii=True)

from ..dataloader import GeneData
from .loss import entropy_regularization, independence_loss
from .loss import reference_consistency_loss


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def FT_Trainer(
    train_loader,
    model,
    optimizer,
    ssl_loss,
    tsk_loss,
    device,
    alpha=0.0,
    correction=0.0,
    entropy_weight=0.0,
):

    model.train()
    total_loss = []
    total_ssl_loss = []
    total_tsk_loss = []

    # torch.autograd.set_detect_anomaly(True)
    # for data in tqdm(train_loader, ascii=True):
    for data in train_loader:

        triplet, label = data

        anchor_y_true, positive_y_true, negative_y_true = label
        anchor, positive, negative = triplet

        anchor, positive, negative = (
            anchor.to(device),
            positive.to(device),
            negative.to(device),
        )
        anchor_y_true = anchor_y_true.to(device)
        positive_y_true = positive_y_true.to(device)
        negative_y_true = negative_y_true.to(device)

        optimizer.zero_grad()

        (anchor_emb, anchor_refg), anchor_y_pred = model(anchor)
        (positive_emb, positive_refg), positive_y_pred = model(positive)
        (negative_emb, negative_refg), negative_y_pred = model(negative)

        lss = ssl_loss(anchor_emb, positive_emb, negative_emb)

        ## remove batch effects by minimal the differences between house-keeping genes
        if correction != 0:
            ref = reference_consistency_loss(
                anchor_refg[1], positive_refg[1], negative_refg[1]
            )
            lss = (1 - correction) * lss + correction * ref
            # print("Ref: {:.6f} - lss: {:.2f}".format(ref.item(), lss.item()))

        y_pred = anchor_y_pred  # torch.cat([anchor_y_pred, positive_y_pred, negative_y_pred])
        y_true = anchor_y_true  # torch.cat([anchor_y_true, positive_y_true, negative_y_true])
        tsk = tsk_loss(y_pred, y_true)

        if entropy_weight != 0:
            entropy_reg = entropy_regularization(y_pred)
            tsk = tsk * (1 - entropy_weight) + entropy_reg * entropy_weight

        loss = (1.0 - alpha) * lss + tsk * alpha

        # print(cv_loss, lss, loss)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        total_ssl_loss.append(lss.item())
        total_tsk_loss.append(tsk.item())

    train_total_loss = np.mean(total_loss)
    train_ssl_loss = np.mean(total_ssl_loss)
    train_tsk_loss = np.mean(total_tsk_loss)

    return train_total_loss, train_ssl_loss, train_tsk_loss


@torch.no_grad()
def FT_Tester(test_loader, model, ssl_loss, tsk_loss, device, alpha=0.0, correction=0):
    model.eval()
    total_loss = []
    total_ssl_loss = []
    total_tsk_loss = []

    for data in test_loader:
        triplet, label = data
        anchor, positive, negative = triplet
        anchor_y_true, positive_y_true, negative_y_true = label

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        anchor_y_true = anchor_y_true.to(device)
        positive_y_true = positive_y_true.to(device)
        negative_y_true = negative_y_true.to(device)

        (anchor_emb, anchor_refg), anchor_y_pred = model(anchor)

        (positive_emb, positive_refg), positive_y_pred = model(positive)
        (negative_emb, negative_refg), negative_y_pred = model(negative)

        lss = ssl_loss(anchor_emb, positive_emb, negative_emb)

        if correction != 0:
            ref = reference_consistency_loss(
                anchor_refg[1], positive_refg[1], negative_refg[1]
            )
            lss = (1 - correction) * lss + correction * ref
            # print("Ref: {:.6f} - lss: {:.2f}".format(ref.item(), lss.item()))

        # torch.cat([anchor_y_pred, positive_y_pred, negative_y_pred], dim = 0)
        y_pred = anchor_y_pred
        y_true = anchor_y_true
        tsk = tsk_loss(y_pred, y_true)

        loss = (1.0 - alpha) * lss + tsk * alpha

        total_loss.append(loss.item())
        total_ssl_loss.append(lss.item())
        total_tsk_loss.append(tsk.item())

    test_total_loss = np.mean(total_loss)
    test_ssl_loss = np.mean(total_ssl_loss)
    test_tsk_loss = np.mean(total_tsk_loss)

    return test_total_loss, test_ssl_loss, test_tsk_loss


from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as prc_auc_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


def scorer(y_true, y_pred):

    y_prob = y_pred[:, 1]
    y_pred = y_pred.argmax(axis=1)
    y_true = y_true.argmax(axis=1)
    if len(np.unique(y_true)) == 1:
        roc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
    _precision, _recall, _ = precision_recall_curve(y_true, y_prob)
    prc = prc_auc_score(_recall, _precision)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return f1, mcc, prc, roc, acc


@torch.no_grad()
def Evaluator(test_loader, model, device):
    model.eval()
    y_trues = []
    y_preds = []
    for data in test_loader:
        triplet, label = data
        anchor, positive, negative = triplet
        anchor_y_true, positive_y_true, negative_y_true = label

        anchor = anchor.to(device)
        anchor_y_true = anchor_y_true.to(device)

        (anchor_emb, anchor_refg), anchor_y_pred = model(anchor)
        anchor_y_pred = torch.nn.functional.softmax(anchor_y_pred, dim=1)

        y_trues.append(anchor_y_true)
        y_preds.append(anchor_y_pred)

    y_true = torch.concat(y_trues, axis=0).cpu().detach().numpy()
    y_pred = torch.concat(y_preds, axis=0).cpu().detach().numpy()

    f1, mcc, prc, roc, acc = scorer(y_true, y_pred)
    return f1, mcc, prc, roc, acc


@torch.no_grad()
def Predictor(dfcx, model, scaler, device="cpu", batch_size=512, num_workers=4):
    model.eval()
    dfcx = scaler.transform(dfcx)

    predict_tcga = GeneData(dfcx)
    predict_loader = Torchdata.DataLoader(
        predict_tcga,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )
    embds = []
    ys = []
    for anchor in tqdm(predict_loader, ascii=True):
        anchor = anchor.to(device)
        (anchor_emb, anchor_refg), anchor_ys = model(anchor)
        anchor_ys = torch.nn.functional.softmax(anchor_ys, dim=1)
        embds.append(anchor_emb)
        ys.append(anchor_ys)

    embeddings = torch.concat(embds, axis=0).cpu().detach().numpy()
    predictions = torch.concat(ys, axis=0).cpu().detach().numpy()

    if len(model.embed_feature_names) == embeddings.shape[1]:
        columns = model.embed_feature_names
    else:
        columns = model.proj_feature_names

    dfe = pd.DataFrame(embeddings, index=predict_tcga.patient_name, columns=columns)

    ref_cols = model.latentprojector.GENESET.iloc[
        model.latentprojector.CELLPATHWAY.loc["Reference"]
    ].index.tolist()
    ref_cols.append("Reference")

    if not model.ref_for_task:
        dfe = dfe[dfe.columns.difference(ref_cols)]

    dfp = pd.DataFrame(predictions, index=predict_tcga.patient_name)

    return dfe, dfp


@torch.no_grad()
def Extractor(
    dfcx,
    model,
    scaler,
    device="cpu",
    batch_size=512,
    num_workers=4,
    with_gene_level=False,
):
    """
    Extract geneset-level and celltype-level features
    """
    model.eval()
    dfcx = scaler.transform(dfcx)
    genesetprojector = model.latentprojector.genesetprojector
    cellpathwayprojector = model.latentprojector.cellpathwayprojector

    predict_tcga = GeneData(dfcx)
    predict_loader = Torchdata.DataLoader(
        predict_tcga,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )
    geneset_feat = []
    celltype_feat = []
    gene_feat = []

    for anchor in tqdm(predict_loader, ascii=True):
        anchor = anchor.to(device)
        encoding = model.inputencoder(anchor)

        gene_level_proj = genesetprojector.geneset_scorer(encoding)[
            :, 2:
        ]  # remove pid, cancer
        geneset_level_proj, cellpathway_level_proj = model.latentprojector(encoding)

        geneset_feat.append(geneset_level_proj)
        celltype_feat.append(cellpathway_level_proj)
        gene_feat.append(gene_level_proj)

    genefeatures = torch.concat(gene_feat, axis=0).cpu().detach().numpy()
    genesetfeatures = torch.concat(geneset_feat, axis=0).cpu().detach().numpy()
    celltypefeatures = torch.concat(celltype_feat, axis=0).cpu().detach().numpy()

    dfgeneset = pd.DataFrame(
        genesetfeatures,
        index=predict_tcga.patient_name,
        columns=model.geneset_feature_name,
    )
    dfcelltype = pd.DataFrame(
        celltypefeatures,
        index=predict_tcga.patient_name,
        columns=model.celltype_feature_name,
    )

    dfgene = pd.DataFrame(
        genefeatures, index=predict_tcga.patient_name, columns=predict_tcga.feature_name
    )

    return dfgene, dfgeneset, dfcelltype


@torch.no_grad()
def Projector(dfcx, model, scaler, device="cpu", batch_size=64, num_workers=4):
    """
    Extract geneset-level and celltype-level features
    """

    model.eval()
    gs_projector = model.latentprojector.genesetprojector
    ct_projector = model.latentprojector.cellpathwayprojector

    predict_tcga = GeneData(dfcx)
    predict_loader = Torchdata.DataLoader(
        predict_tcga,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )
    geneset_feat = []
    celltype_feat = []

    for anchor in tqdm(predict_loader, ascii=True):
        anchor = anchor.to(device)

        x = model.inputencoder(anchor)
        pid_encoding = x[:, 0:1, :]  # take the learnbale patient id token
        cancer_encoding = x[:, 1:2, :]  # take the cancer_type token
        gene_encoding = x[:, 2:, :]  # take the gene encoding

        geneset_level_proj = gs_projector.geneset_aggregator(x)

        b, f, c = geneset_level_proj.shape

        ct_feats = []
        for i in range(c):
            ct_ = ct_projector(geneset_level_proj[:, :, i])
            ct_ = ct_.cpu().detach().numpy()
            ct_feats.append(ct_)

        celltype_level_proj = np.stack(ct_feats, axis=-1)
        geneset_level_proj = geneset_level_proj.cpu().detach().numpy()

        geneset_feat.append(geneset_level_proj)
        celltype_feat.append(celltype_level_proj)

    gs_feat = np.concatenate(geneset_feat, axis=0)
    ct_feat = np.concatenate(celltype_feat, axis=0)

    b, f, c = gs_feat.shape
    feature_name = model.latentprojector.GENESET.index
    feature_sample_labels = [
        dfcx.index[i // f] + "$$" + feature_name[i % f] for i in range(b * f)
    ]
    dfgs = pd.DataFrame(
        gs_feat.reshape(b * f, c),
        index=feature_sample_labels,
        columns=["channel_%s" % i for i in range(c)],
    )

    b, f, c = ct_feat.shape
    feature_name = model.latentprojector.CELLPATHWAY.index
    index = [dfcx.index[i // f] + "$$" + feature_name[i % f] for i in range(b * f)]
    columns = ["channel_%s" % i for i in range(c)]
    dfct = pd.DataFrame(ct_feat.reshape(b * f, c), index=index, columns=columns)
    return dfgs, dfct
