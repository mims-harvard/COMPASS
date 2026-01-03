# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:05:13 2023

@author: Wanxiang Shen
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, metric="cosine"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        self.name = "TripletLoss"

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:

        if self.metric == "euclidean":
            distance_positive = self.calc_euclidean(anchor, positive)
            distance_negative = self.calc_euclidean(anchor, negative)
        elif self.metric == "cosine":
            distance_positive = 1 - F.cosine_similarity(anchor, positive)
            distance_negative = 1 - F.cosine_similarity(anchor, negative)

        else:
            raise ValueError("Unsupported metric: {}".format(self.metric))

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        # print(distance_positive.mean())

        return losses.mean()


class TriSimplexLoss(nn.Module):

    def __init__(self):
        super(TriSimplexLoss, self).__init__()
        self.name = "TriSimplexLoss"

    def forward(self, trisimplex_input, trisimplex_emb):
        a, p, n = trisimplex_input
        a_emb, p_emb, n_emb = trisimplex_emb
        # s1 = F.cosine_similarity(a, p)
        # s2 = F.cosine_similarity(a, n)
        # s3 = F.cosine_similarity(n, p)
        # s1_emb = F.cosine_similarity(a_emb, p_emb)
        # s2_emb = F.cosine_similarity(a_emb, n_emb)
        # s3_emb = F.cosine_similarity(n_emb, p_emb)
        # loss = F.mse_loss(s1, s1_emb) + F.mse_loss(s2, s2_emb) + F.mse_loss(s3, s3_emb)

        sim_a = self._pairwise_cosine_sim(a)
        sim_a_emb = self._pairwise_cosine_sim(a_emb)

        loss = F.mse_loss(sim_a, sim_a_emb)
        return loss

    def _pairwise_cosine_sim(self, tensor):
        # Normalize each vector in the tensor
        normalized_tensor = tensor / tensor.norm(dim=1, keepdim=True)
        # Perform matrix multiplication with its transpose to get cosine similarity
        similarity_matrix = torch.matmul(normalized_tensor, normalized_tensor.T)

        upper_triangular = torch.triu(similarity_matrix, diagonal=1)
        indices = upper_triangular != 0
        non_squareform_vector = upper_triangular[indices]
        return non_squareform_vector


class MAEWithNaNLabelsLoss(nn.Module):
    def __init__(self):
        super(MAEWithNaNLabelsLoss, self).__init__()

    def forward(self, predictions, labels):
        mask = ~torch.isnan(labels)[:, 0]  # Assuming labels is a 2D tensor
        if mask.any():
            # mae_loss = torch.mean((predictions[mask] - labels[mask]) ** 2)
            mae_loss = torch.mean(torch.abs(predictions[mask] - labels[mask]))

            return mae_loss
        else:
            return torch.tensor(0.0, device=labels.device, dtype=labels.dtype)



class CEWithNaNLabelsLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, logits, labels):
        """
        logits: (B, C) raw logits
        labels: (B, C) one-hot / soft labels, may contain NaN
        """
        # mask rows with any NaN
        mask = ~torch.isnan(labels).any(dim=1)
        if not mask.any():
            return torch.tensor(0.0, device=labels.device)

        y_pred = logits[mask]                 # (B, C)
        y_true = labels[mask].argmax(dim=1)   # (B,)

        if self.weights is None:
            return F.cross_entropy(y_pred, y_true)
        else:
            w = torch.tensor(
                self.weights,
                device=y_true.device,
                dtype=torch.float,
            )
            return F.cross_entropy(y_pred, y_true, weight=w)




class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al., 2020)."""

    def __init__(self, weights=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        alpha = weights
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        if isinstance(alpha, (float, int, list)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, predictions, labels):
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
        pt = torch.gather(predictions, 1, labels.unsqueeze(1)).squeeze()
        if self.alpha is not None:
            if self.alpha.type() != predictions.data.type():
                self.alpha = self.alpha.type_as(predictions.data)
            at = torch.gather(self.alpha, 0, labels)
        else:
            at = 1.0
        log_pt = torch.log(pt)
        loss = -at * ((1 - pt) ** self.gamma) * log_pt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        dice_coeff = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1 - dice_coeff
        return dice_loss


class DSCLoss(nn.Module):
    def __init__(
        self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean"
    ):
        super(DSCLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, probs, targets):
        targets = targets.float()
        intersection = (probs * targets).sum(dim=1)
        weighted_intersection = (2.0 * intersection + self.smooth) ** self.alpha

        cardinality = (probs + targets).sum(dim=1)
        weighted_cardinality = (cardinality + self.smooth) ** self.alpha
        dice_coeff = weighted_intersection / weighted_cardinality
        dice_loss = 1 - dice_coeff
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, y_pred, y_true):

        # convert to range: [-1,1]
        y_true_transformed = 2 * y_true - 1
        y_pred_transformed = 2 * y_pred - 1
        loss = F.relu(1 - y_true_transformed * y_pred_transformed)
        return torch.mean(loss)

def entropy_regularization(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(
        probs * torch.log(probs.clamp(min=1e-8)), dim=1
    )
    return entropy.mean()


def reference_consistency_loss(a, p, n, cross_triplet=True):
    # Stack the tensors to create a [b, 3] tensor for batch operation
    """
    To minimize the differences in representations of reference () genes across triplet.
    """
    if cross_triplet:
        reference_representations = torch.stack(
            (a.squeeze(), p.squeeze(), n.squeeze()), dim=1
        )
        loss = cv_loss(reference_representations, dim=1)
    else:
        reference_representations = torch.cat([a, p, n], dim=0)
        loss = cv_loss(reference_representations, dim=0)

    return loss


def cv_loss(reference_representations, dim=0):
    """
    To minimize the differences in representations of reference () genes across samples.
    Reasons: ubiquitously expressed genes are refer to genes that are consistently expressed across different samples, tissues, cell types,
    and conditions. Similar to housekeeping genes, ubiquitously expressed genes provide essential cellular functions necessary for the maintenance
    and survival of cells. They are expressed at stable levels, making them useful as reference points or controls in gene expression studies,
    including quantitative PCR and RNA sequencing analyses, to ensure accurate normalization and comparison across samples.

    While absolute expression levels might not match perfectly across all individuals due to biological variability,
    technical factors, and sensitivity of measurement techniques,
    the expression levels of UEGs should be relatively consistent and close to each other among different people.
    This consistency makes UEGs reliable for comparative studies and normalization in gene expression analyses.
    """

    # print(reference_representations)
    mean_ref = torch.mean(reference_representations, dim=dim)
    std_ref = torch.std(reference_representations, dim=dim)
    # 使用平均值的绝对值
    cv = std_ref / (torch.abs(mean_ref) + 1e-6)
    cv_loss = cv.mean()

    return cv_loss


def cv_loss_penalty(reference_representations, dim=0):
    """
    To minimize the differences in representations of reference (UEGs) genes across samples.
    Reasons: ubiquitously expressed genes (UEGs) are consistently expressed across different samples, tissues, cell types, and conditions. Similar to housekeeping genes, UEGs provide essential cellular functions necessary for the maintenance and survival of cells. They are expressed at stable levels, making them useful as reference points or controls in gene expression studies, including quantitative PCR and RNA sequencing analyses, to ensure accurate normalization and comparison across samples.
    While absolute expression levels might not match perfectly across all individuals due to biological variability, technical factors, and sensitivity of measurement techniques, the expression levels of UEGs should be relatively consistent and close to each other among different people.
    This consistency makes UEGs reliable for comparative studies and normalization in gene expression analyses.
    """

    # Ensure the input is positive by adding a penalty for negative values
    penalty = torch.nn.ReLU()(1e-6 - reference_representations).sum()

    # Calculate the mean and standard deviation
    mean_ref = torch.mean(reference_representations, dim=dim)
    std_ref = torch.std(reference_representations, dim=dim)

    # Calculate the coefficient of variation
    cv = std_ref / (torch.abs(mean_ref) + 1e-6)
    cv_loss = cv.mean()

    # Combine cv_loss with the penalty
    total_loss = cv_loss + penalty

    return total_loss


def msd_loss(expression_levels, dim=0, target_mean=1.0, regularization_weight=0.1):
    """
    Minimize variance in expression levels of reference genes across samples, while encouraging expression levels to be close to target_mean.

    Parameters:
    expression_levels (torch.Tensor): Tensor of shape [num_samples, num_genes]
                                      representing expression levels of genes across samples.
    dim (int): Dimension along which to calculate the mean.
    target_mean (float): Target mean expression level.
    regularization_weight (float): Weight of the regularization term.

    Returns:
    torch.Tensor: Scalar loss value.
    """
    # Calculate the mean expression level for each gene across samples
    mean_expression = torch.mean(expression_levels, dim=dim, keepdim=True)

    # Calculate the abs difference from the target mean for each gene
    diffs = torch.abs(expression_levels - target_mean)

    # Calculate the regularization term
    reg_term = torch.mean((expression_levels - target_mean) ** 2)

    # Average the absolute differences across all genes and samples
    loss = torch.mean(diffs) + regularization_weight * reg_term

    return loss


def independence_loss(x, y):
    """
    a loss function that penalizes correlation between x and y.
    """
    y_binary = torch.argmax(y, dim=1).float()
    y_mean = torch.mean(y_binary)
    x_mean = torch.mean(x)
    x_std = torch.std(x, unbiased=True) + 1e-8
    n_pos = torch.sum(y_binary)
    n_neg = len(y_binary) - n_pos
    x_mean_pos = torch.mean(x[y_binary == 1])
    x_mean_neg = torch.mean(x[y_binary == 0])
    pbc = (x_mean_pos - x_mean_neg) * torch.sqrt(n_pos * n_neg) / (x_std * len(x))
    return torch.abs(pbc)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(
            similarity_matrix / self.temperature
        )

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss
