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



import os
def _remap_distilled_encoder(pretrainer):
    """
    The original model was trained with FlashAttention and a batch size of 1024, 
    which makes exact reproduction hardware-dependent. To improve reproducibility, 
    we distilled the model onto a Performer encoder and remapped its weights, 
    achieving similar accuracy with better portability.

    Parameters
    ----------
    pretrainer : object
        A COMPASS pretrainer object with attributes:
        - pretrainer.model.inputencoder: the encoder to update
        - pretrainer.feature_name: list of gene names for the new model

    Returns
    -------
    updated_state_dict : dict
        A new inputencoder state_dict with aligned and remapped weights.
    """

    # --- 1. Load distilled encoder weights ---
    distilled_path = os.path.join(os.path.dirname(__file__), "_distilled.pth")
    # if not os.path.exists(distilled_path):
    #     raise FileNotFoundError(f"Distilled checkpoint not found: {distilled_path}")

    full_encoder_sd, full_feature_list = torch.load(distilled_path, map_location=pretrainer.device)

    new_encoder_sd = pretrainer.model.inputencoder.state_dict()
    new_feature_list = list(pretrainer.feature_name)

    # --- 2. Build gene index mapping ---
    old_gene_to_idx = {gene: idx for idx, gene in enumerate(full_feature_list)}
    shared_genes = [g for g in new_feature_list if g in old_gene_to_idx]

    shared_idx_old = np.array([old_gene_to_idx[g] for g in shared_genes], dtype=int)
    shared_idx_new = np.array(
        [i for i, g in enumerate(new_feature_list) if g in old_gene_to_idx],
        dtype=int,
    )

    #print(f"✅ Shared genes: {len(shared_genes)} / {len(new_feature_list)} "
    #      f"({len(shared_genes)/len(new_feature_list):.1%})")

    # --- 3. Copy over all matching layers except abundance embedder ---
    updated_sd = new_encoder_sd.copy()
    for key, val in full_encoder_sd.items():
        if "gene_token_embedder.abundance_embedder.layers.0" in key:
            continue
        if key in updated_sd and updated_sd[key].shape == val.shape:
            updated_sd[key] = val.clone()

    # --- 4. Handle abundance embedder (gene-specific layers) ---
    w_key = "gene_token_embedder.abundance_embedder.layers.0.weight"
    b_key = "gene_token_embedder.abundance_embedder.layers.0.bias"

    old_w = full_encoder_sd[w_key]
    old_b = full_encoder_sd[b_key]
    new_w = new_encoder_sd[w_key].clone()
    new_b = new_encoder_sd[b_key].clone()

    with torch.no_grad():
        # Copy shared genes
        new_w[shared_idx_new] = old_w[shared_idx_old]
        new_b[shared_idx_new] = old_b[shared_idx_old]

        # Randomly initialize unmatched genes
        unmatched = list(set(range(len(new_feature_list))) - set(shared_idx_new))
        if unmatched:
            print(f"⚠️ Randomly initializing {len(unmatched)} unmatched genes.")
            new_w[unmatched] = torch.randn(len(unmatched), new_w.shape[1]) * 0.02
            new_b[unmatched] = torch.zeros(len(unmatched), new_b.shape[1])

        updated_sd[w_key] = new_w
        updated_sd[b_key] = new_b

    # --- 5. Done ---
    #print("✅ Distilled encoder remapping complete.")
    pretrainer.model.inputencoder.load_state_dict(updated_sd)
    return pretrainer


