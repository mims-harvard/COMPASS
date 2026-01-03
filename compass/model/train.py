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
from .loss import cv_loss, msd_loss, cv_loss_penalty


import gc


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def PT_Trainer(
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
            refe = torch.cat(
                [anchor_refg[1], positive_refg[1], negative_refg[1]], axis=0
            )
            refy = torch.cat([anchor_y_true, positive_y_true, negative_y_true], axis=0)
            # ref = cv_loss_penalty(refe)
            ref = msd_loss(refe)
            # idp = independence_loss(refe, refy)
            # print("Lss: {:.3f} - Ref: {:.3f} - idp: {:.6f}".format(lss.item(), ref.item(), idp.item()))
            lss = (1 - correction) * lss + correction * ref
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

        torch.cuda.empty_cache()
        gc.collect()

    train_total_loss = np.mean(total_loss)
    train_ssl_loss = np.mean(total_ssl_loss)
    train_tsk_loss = np.mean(total_tsk_loss)

    return train_total_loss, train_ssl_loss, train_tsk_loss


@torch.no_grad()
def PT_Tester(test_loader, model, ssl_loss, tsk_loss, device, alpha=0.0, correction=0):
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
            refe = torch.cat(
                [anchor_refg[1], positive_refg[1], negative_refg[1]], axis=0
            )
            refy = torch.cat([anchor_y_true, positive_y_true, negative_y_true], axis=0)
            # ref = cv_loss(refe) + msd_loss(refe)
            ref = msd_loss(refe)  #
            lss = (1 - correction) * lss + correction * ref
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
