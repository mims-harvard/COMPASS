# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""
import torch
import torch.nn.functional as F
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


def r2_loss(output, target):
    """
    Computes the 1 - R^2 loss between the output and target tensors.

    Args:
        output (torch.Tensor): The predicted values by the model.
        target (torch.Tensor): The actual values.

    Returns:
        torch.Tensor: The 1 - R^2 loss.
    """
    output_mean = torch.mean(output)
    target_mean = torch.mean(target)
    cov = torch.sum((output - output_mean) * (target - target_mean))
    output_std = torch.sqrt(torch.sum((output - output_mean) ** 2))
    target_std = torch.sqrt(torch.sum((target - target_mean) ** 2))

    r = cov / (output_std * target_std)
    return 1 - r**2


def Adp_Trainer(train_loader, model, optimizer, tsk_loss, device, ctp_idx):

    model.train()
    total_loss = []

    # torch.autograd.set_detect_anomaly(True)
    # for data in tqdm(train_loader, ascii=True):
    for data in train_loader:

        triplet, label = data

        anchor_y_true, positive_y_true, negative_y_true = label
        anchor, positive, negative = triplet

        anchor = anchor.to(device)
        anchor_y_true = anchor_y_true.to(device)

        optimizer.zero_grad()

        (anchor_emb, anchor_refg), _ = model(anchor)

        y_pred = anchor_emb[:, [ctp_idx]]
        y_true = anchor_y_true  # torch.cat([anchor_y_true, positive_y_true, negative_y_true])

        # print(y_pred.shape, y_true.shape)

        loss = F.l1_loss(y_pred, y_true)

        # print(loss)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    train_total_loss = np.mean(total_loss)

    return train_total_loss


@torch.no_grad()
def Adp_Tester(test_loader, model, optimizer, tsk_loss, device, ctp_idx):

    model.eval()
    total_loss = []

    for data in test_loader:
        triplet, label = data
        anchor, positive, negative = triplet
        anchor_y_true, positive_y_true, negative_y_true = label

        anchor = anchor.to(device)
        anchor_y_true = anchor_y_true.to(device)
        (anchor_emb, anchor_refg), _ = model(anchor)

        y_pred = anchor_emb[:, [ctp_idx]]
        y_true = anchor_y_true
        loss = F.l1_loss(y_pred, y_true)

        # print(y_pred, y_true)

        total_loss.append(loss.item())

    test_total_loss = np.mean(total_loss)

    return test_total_loss

