# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", font_scale=1.5)
from tqdm import tqdm
import torch
import wandb

import torch.utils.data as data
from copy import deepcopy
from collections import OrderedDict
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold

import random
import datetime


from compass.dataloader import TCGAData, GeneData, ITRPData
from compass.augmentor import (
    RandomMaskAugmentor,
    FeatureJitterAugmentor,
    MaskJitterAugmentor,
)
from compass.model.scaler import Datascaler
from compass.model.model import Compass
from compass.model.train import PT_Trainer, PT_Tester
from compass.model.adapt import Adp_Trainer, Adp_Tester
from compass.model.adapt import _remap_distilled_encoder

from compass.model.tune import (
    FT_Trainer,
    FT_Tester,
    Predictor,
    Evaluator,
    Extractor,
    Projector,
)

from compass.model.loss import TripletLoss, CEWithNaNLabelsLoss, MAEWithNaNLabelsLoss
from compass.model.loss import FocalLoss, DiceLoss, DSCLoss, HingeLoss
from compass.model.saver import SaveBestModel
from compass.utils import (
    plot_embed_with_label,
    loadcompass,
)  #  code for compass model loading

def fixseed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# fixseed(seed=42)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


class PreTrainer:

    def __init__(
        self,
        device="cuda",
        lr=1e-5,
        weight_decay=1e-6,
        epochs=100,
        patience=10,
        batch_size=64,
        embed_dim=32,
        task_loss_weight=0.0,
        task_dense_layer=[24],
        task_batch_norms=True,
        task_class_weight=None,
        encoder="transformer",
        encoder_dropout=0.2,
        num_cancer_types=33,
        transformer_dim=32,
        transformer_num_layers=1,
        transformer_nhead=2,
        transformer_pos_emb="learnable",
        batch_correction=0.0,
        proj_level="cellpathway",
        proj_pid=False,
        proj_cancer_type=True,
        ref_for_task=True,
        triplet_metric="cosine",
        triplet_margin=1.0,
        K=1,
        seed=42,
        work_dir="./results",
        verbose=True,
        with_wandb=False,
        wandb_project="pretrain",
        wandb_dir="./wandb/",
        wandb_entity="senwanxiang",
        **encoder_kwargs,
    ):
        """
        transformer_pos_emb: {None, 'umap', 'pumap', 'gene2vect'}
        encoder:{'mlp', 'transformer'}
        """

        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.num_cancer_types = num_cancer_types

        self.triplet_margin = triplet_margin
        self.triplet_metric = triplet_metric

        self.K = K

        self.task_loss_weight = task_loss_weight
        self.task_dense_layer = task_dense_layer
        self.task_batch_norms = task_batch_norms
        self.task_class_weight = task_class_weight
        self.encoder = encoder
        self.encoder_dropout = encoder_dropout
        self.transformer_dim = transformer_dim
        self.transformer_nhead = transformer_nhead
        self.transformer_num_layers = transformer_num_layers
        self.transformer_pos_emb = transformer_pos_emb

        self.batch_correction = batch_correction
        self.proj_level = proj_level
        self.proj_pid = proj_pid
        self.proj_cancer_type = proj_cancer_type
        self.ref_for_task = ref_for_task

        fixseed(seed=seed)
        self.seed = seed
        self.patience = patience  # for early stopping

        self.work_dir = work_dir
        self.with_wandb = with_wandb
        self.verbose = verbose
        self.wandb_project = wandb_project
        self.wandb_dir = wandb_dir
        self.wandb_entity = wandb_entity
        self.encoder_kwargs = encoder_kwargs

    def _setup(self, input_dim, task_dim, task_type, save_dir, run_name):

        model = Compass(
            input_dim,
            task_dim,
            task_type,
            proj_level=self.proj_level,
            proj_pid=self.proj_pid,
            proj_cancer_type=self.proj_cancer_type,
            ref_for_task=self.ref_for_task,
            num_cancer_types=self.num_cancer_types,
            embed_dim=self.embed_dim,
            encoder=self.encoder,
            encoder_dropout=self.encoder_dropout,
            transformer_dim=self.transformer_dim,
            transformer_nhead=self.transformer_nhead,
            transformer_num_layers=self.transformer_num_layers,
            transformer_pos_emb=self.transformer_pos_emb,
            task_dense_layer=self.task_dense_layer,
            task_batch_norms=self.task_batch_norms,
            seed=self.seed,
            **self.encoder_kwargs,
        )

        model = model.to(self.device)
        self.model = model
        
        ssl_loss = TripletLoss(margin=self.triplet_margin, metric=self.triplet_metric)
        ce_loss = CEWithNaNLabelsLoss(weights=self.task_class_weight)
        mae_loss = MAEWithNaNLabelsLoss()

        _remap_distilled_encoder(self) # knowledge-distilled from the FlashAttention-based weights
        plist = [
                {"params": model.inputencoder.parameters(), 'lr': 1e-8},
                {"params": model.latentprojector.parameters()},  
                {"params": model.taskdecoder.parameters()},  
                ]

        optimizer = torch.optim.Adam(plist, lr=self.lr, weight_decay=self.weight_decay)
        saver = SaveBestModel(save_dir=save_dir, save_name="model.pth")

        if task_type == "c":
            tsk_loss = ce_loss
        elif task_type == "r":
            tsk_loss = mae_loss
        else:
            raise ValueError("Invalid task_type. Use 'c', or 'r'. ")

        self.ssl_loss = ssl_loss
        self.tsk_loss = tsk_loss
        self.optimizer = optimizer
        self.saver = saver

        if self.with_wandb:
            # Initialize wandb
            self.wandb = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                dir=self.wandb_dir,
                name=run_name,
                save_code=True,
            )

            # # Log model architecture to wandb
            self.wandb.watch(model)

    def train(
        self,
        dfcx_train,
        dfy_train=None,
        dfcx_test=None,
        dfy_test=None,
        task_name="notask",
        task_type="c",
        aug_method="mix",
        scale_method="minmax",
        **augargs,
    ):

        ### scaler ####
        self.scale_method = scale_method
        self.scaler = Datascaler(scale_method=scale_method)
        self.scaler = self.scaler.fit(dfcx_train)
        self.aug_method = aug_method

        dfcx_train = self.scaler.transform(dfcx_train)

        if aug_method == "mask":
            self.augmentor = RandomMaskAugmentor(**augargs)
        elif aug_method == "jitter":
            self.augmentor = FeatureJitterAugmentor(**augargs)
        elif aug_method == "mix":
            self.augmentor = MaskJitterAugmentor(**augargs)
        else:
            raise ValueError("Invalid method. Use 'mask', 'jitter' or 'mix'.")

        self.task_type = task_type
        self.task_name = task_name

        if dfy_train is None:
            dfy_train = pd.DataFrame(
                np.zeros((len(dfcx_train), 2)), index=dfcx_train.index
            )

        train_tcga = TCGAData(dfcx_train, dfy_train, self.augmentor, K=self.K)

        self.y_scaler = train_tcga.y_scaler
        self.feature_name = train_tcga.feature_name

        train_loader = data.DataLoader(
            train_tcga,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )

        input_dim = len(train_tcga.feature_name)
        task_dim = train_tcga.y.shape[1]

        if dfcx_test is not None:
            dfcx_test = self.scaler.transform(dfcx_test)
            if dfy_test is None:
                dfy_test = pd.DataFrame(
                    np.zeros((len(dfcx_test), 2)), index=dfcx_test.index
                )
            test_tcga = TCGAData(dfcx_test, dfy_test, self.augmentor, K=self.K)
            test_loader = data.DataLoader(
                test_tcga,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=4,
            )
        else:
            test_loader = None

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime(
            "%Y%m%d_%H%M%S"
        )  # Format: YYYYMMDD_HHMMSS
        run_name = f"Pretrain_{self.task_name}_{formatted_time}"  # Example filename

        self.run_name = run_name
        self.save_dir = os.path.join(self.work_dir, run_name)

        ## init model, operimizer, ...
        self._setup(input_dim, task_dim, task_type, self.save_dir, self.run_name)
        self.input_dim = input_dim
        self.task_dim = task_dim

        ### training ###
        performace = []
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.epochs):
            train_total_loss, train_ssl_loss, train_tsk_loss = PT_Trainer(
                train_loader,
                self.model,
                self.optimizer,
                self.ssl_loss,
                self.tsk_loss,
                self.device,
                alpha=self.task_loss_weight,
                correction=self.batch_correction,
            )

            if test_loader is not None:
                test_total_loss, test_ssl_loss, test_tsk_loss = PT_Tester(
                    test_loader,
                    self.model,
                    self.ssl_loss,
                    self.tsk_loss,
                    self.device,
                    alpha=self.task_loss_weight,
                    correction=self.batch_correction,
                )

                self.saver(
                    test_total_loss, epoch, self.model, self.optimizer, self.scaler
                )

                # Early Stopping Check
                if test_total_loss < best_val_loss:
                    best_val_loss = test_total_loss
                    patience_counter = 0  # Reset the counter if improvement is seen
                else:
                    patience_counter += 1  # Increment the counter if no improvement

            else:
                test_total_loss, test_ssl_loss, test_tsk_loss = np.nan, np.nan, np.nan
                self.saver(
                    train_total_loss, epoch, self.model, self.optimizer, self.scaler
                )

                # Early Stopping Check
                if train_total_loss < best_val_loss:
                    best_val_loss = train_total_loss
                    patience_counter = 0  # Reset the counter if improvement is seen
                else:
                    patience_counter += 1  # Increment the counter if no improvement

            performace.append(
                [
                    epoch,
                    train_total_loss,
                    train_ssl_loss,
                    train_tsk_loss,
                    test_total_loss,
                    test_ssl_loss,
                    test_tsk_loss,
                ]
            )

            if self.with_wandb:
                self.wandb.log(
                    {
                        "pretrain_loss": train_total_loss,
                        "pretrain_ssl_loss": train_ssl_loss,
                        "pretrain_%s_loss" % self.task_name: train_tsk_loss,
                        "pretrain_test_loss": test_total_loss,
                        "pretrain_test_ssl_loss": test_ssl_loss,
                        "pretrain_test_%s_loss" % self.task_name: test_tsk_loss,
                    }
                )
            if self.verbose:
                print(
                    "Epoch: {}/{} - Train Loss: {:.4f} - Test Loss: {:.4f}".format(
                        epoch + 1, self.epochs, train_total_loss, test_total_loss
                    )
                )

            # for early stopping
            if patience_counter >= self.patience:
                print(
                    f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {self.patience} consecutive epochs."
                )
                break

        self.saver.save()
        self.performace = performace

        ## plot loss locally
        df = pd.DataFrame(
            self.performace,
            columns=[
                "epochs",
                "total_loss",
                "ssl_loss",
                "%s_loss" % self.task_name,
                "test_loss",
                "test_ssl_loss",
                "test_%s_loss" % self.task_name,
            ],
        ).set_index("epochs")
        # v = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
        fig, ax = plt.subplots(figsize=(7, 5))
        df[["ssl_loss", "test_ssl_loss"]].plot(ax=ax)
        fig.savefig(
            os.path.join(self.save_dir, "tcga_train_loss.png"), bbox_inches="tight"
        )
        df.to_pickle(os.path.join(self.save_dir, "tcga_train_loss.pkl"))

        return self

    def plot_embed(self, df_tpm, df_label, label_types, **kwargs):
        dfe, dfp = self.predict(df_tpm, batch_size=self.batch_size, num_workers=4)
        dfd = dfe.join(df_label)
        label_cols = df_label.columns

        figs = plot_embed_with_label(
            dfd, label_col=label_cols, label_type=label_types, **kwargs
        )
        for fig, name in zip(figs, label_cols):
            fig.savefig(
                os.path.join(self.save_dir, "tcga_%s.png" % name),
                bbox_inches="tight",
            )

    def predict(self, df_tpm, batch_size=512, num_workers=4):
        model = Compass(**self.saver.inMemorySave["model_args"])
        model.load_state_dict(self.saver.inMemorySave["model_state_dict"], strict=False)
        model = model.to(self.device)
        dfe, dfp = Predictor(
            df_tpm,
            model,
            self.scaler,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return dfe, dfp

    def extract(self, df_tpm, batch_size=512, num_workers=4, with_gene_level=False):
        model = Compass(**self.saver.inMemorySave["model_args"])
        model.load_state_dict(self.saver.inMemorySave["model_state_dict"], strict=False)
        model = model.to(self.device)
        dfg, dfgs, dfct = Extractor(
            df_tpm,
            model,
            self.scaler,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
            with_gene_level=True,
        )
        if with_gene_level:
            return dfg, dfgs, dfct
        else:
            return dfgs, dfct

    def project(self, df_tpm, batch_size=512, num_workers=4):
        model = Compass(**self.saver.inMemorySave["model_args"])
        model.load_state_dict(self.saver.inMemorySave["model_state_dict"], strict=False)
        model = model.to(self.device)
        dfg, dfc = Projector(
            df_tpm,
            model,
            self.scaler,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return dfg, dfc

    def save(self, mfile):
        if self.with_wandb:
            self.wandb._settings = ""
        torch.save(self, mfile)
        print("Saving the model to %s" % mfile)

    def load(self, mfile, **kwargs):
        self = loadcompass(mfile, **kwargs)
        if self.with_wandb:
            self.wandb._settings = ""
        return self

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def close(self):
        if self.with_wandb:
            self.wandb.finish()
            self.wandb._settings = ""
        self.save(os.path.join(self.save_dir, "pretrainer.pt"))

    def copy(self):
        return deepcopy(self)


class Adapter:

    def __init__(
        self,
        pretrainer,
        adp_feature="TMB",
        lr=1e-3,
        weight_decay=1e-6,
        batch_size=128,
        epochs=100,
        patience=10,
        save_dir="./",
        verbose=True,
        with_wandb=False,
        save_best_model=False,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.pretrainer = pretrainer.copy()
        self.adp_feature = adp_feature
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.with_wandb = with_wandb

        self._init_adaptive_model(Key=self.adp_feature)
        self.verbose = verbose
        self.save_best_model = save_best_model
        self.scaler = pretrainer.scaler
        self.device = pretrainer.device

    def _init_adaptive_model(self, Key="TMB"):

        fixseed(self.pretrainer.seed)  # Resets the random seed
        model_args = deepcopy(self.pretrainer.saver.inMemorySave["model_args"])
        model_weights = deepcopy(self.pretrainer.saver.inMemorySave["model_state_dict"])

        ### define finetune model
        model = Compass(**model_args)

        encoder_state = OrderedDict()
        for k, v in model_weights.items():
            encoder_state[k] = v

        ### load Pretrained model
        model.load_state_dict(encoder_state, strict=False)
        model = model.to(self.pretrainer.device)

        ctp_idx = (
            model.latentprojector.cellpathwayprojector.CELLPATHWAY.index.tolist().index(
                Key
            )
        )
        proj_idx = model.latentprojector.cellpathway_proj_cols.index(Key)

        ctp_name = "cellpathway_%s" % ctp_idx

        gs_idx_list = model.latentprojector.cellpathwayprojector.CELLPATHWAY.loc[Key]
        gs_name_list = ["geneset_%s" % idx for idx in gs_idx_list]

        for key, param in model.latentprojector.cellpathwayprojector.named_parameters():
            key_idx = key.split(".")[-1]
            if key_idx == ctp_name:
                print(key)
                param.requires_grad = True
            else:
                param.requires_grad = False

        for key, param in model.latentprojector.genesetprojector.named_parameters():
            key_idx = key.split(".")[-1]
            if key_idx in gs_name_list:
                print(key)
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in model.inputencoder.parameters():
            param.requires_grad = False

        for param in model.latentprojector.patientprojector.parameters():
            param.requires_grad = False

        for param in model.latentprojector.cancerprojector.parameters():
            param.requires_grad = False

        for param in model.taskdecoder.parameters():
            param.requires_grad = False

        plist = []
        for param in model.parameters():
            if param.requires_grad:
                plist.append({"params": param})

        optimizer = torch.optim.Adam(plist, lr=self.lr, weight_decay=self.weight_decay)
        loss = MAEWithNaNLabelsLoss()
        saver = SaveBestModel(save_dir=self.save_dir, save_name="adp_model.pth")

        self.model = model
        self.proj_idx = proj_idx
        self.optimizer = optimizer
        self.loss = loss
        self.saver = saver
        plist.append({"params": param})

    def adapt(self, dfcx_train, dfy_train, dfcx_test=None, dfy_test=None):

        dfcx_train = self.scaler.transform(dfcx_train)

        augmentor = RandomMaskAugmentor(no_augment_prob=1)  # no aug

        train_itrp = TCGAData(dfcx_train, dfy_train, augmentor, K=1)
        train_loader = data.DataLoader(
            train_itrp,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            num_workers=4,
        )  #

        if dfcx_test is not None:
            dfcx_test = self.scaler.transform(dfcx_test)
            test_itrp = TCGAData(dfcx_test, dfy_test, augmentor, K=1)
            test_loader = data.DataLoader(
                test_itrp,
                batch_size=self.batch_size,
                shuffle=False,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                num_workers=4,
            )
        else:
            test_loader = None

        ### training ###
        performance = []
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in tqdm(range(self.epochs), ascii=True):

            train_total_loss = Adp_Trainer(
                train_loader,
                self.model,
                self.optimizer,
                self.loss,
                self.device,
                self.proj_idx,
            )

            saving_criteria = train_total_loss

            if test_loader is not None:
                test_total_loss = Adp_Tester(
                    test_loader,
                    self.model,
                    self.optimizer,
                    self.loss,
                    self.device,
                    self.proj_idx,
                )

                saving_criteria = test_total_loss

            else:
                test_total_loss = np.nan

            ## saving_criteria: lower is better
            ## saving model & Early Stopping based on saving_criteria
            self.saver(saving_criteria, epoch, self.model, self.optimizer, self.scaler)
            if saving_criteria < best_val_loss:
                best_val_loss = saving_criteria
                patience_counter = 0
            else:
                patience_counter += 1

            performance.append([epoch, train_total_loss, test_total_loss])

            if self.with_wandb:
                self.wandb.log(
                    {"train_loss": train_total_loss, "test_loss": test_total_loss}
                )

            if self.verbose:
                print(
                    "Epoch: {}/{} - Train Loss: {:.4f} - Test Loss: {:.4f}".format(
                        epoch + 1, self.epochs, train_total_loss, test_total_loss
                    )
                )

            if patience_counter >= self.patience:
                print(
                    f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {self.patience} consecutive epochs."
                )
                break

        if self.save_best_model:
            self.saver.save()
        else:
            os.system('rm -r "%s"' % self.save_dir)

        self.performance = performance
        self.best_epoch = self.saver.inMemorySave["epoch"]

        self.pretrainer.saver = self.saver
        self.pretrainer.model = self.model

        return self.pretrainer

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class FineTuner:
    """
    Few-shot learning (with <full, partial, head> or without FT) on ITRP datasets
    """

    def __init__(
        self,
        pretrainer,
        mode="PFT",
        load_decoder=False,
        device="cuda",
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=500,
        patience=10,
        batch_size=32,
        triplet_metric="cosine",
        triplet_margin=1.0,
        task_loss_weight=1.0,
        task_loss_type="ce_loss",
        task_class_weight=[1, 2],
        task_dense_layer=[16],
        task_batch_norms=True,
        batch_correction=0.0,
        entropy_weight=0.0,
        seed=42,
        verbose=True,
        with_wandb=False,
        wandb_project="finetune",
        wandb_dir="/n/data1/hms/dbmi/zitnik/lab/users/was966/wandb/",
        wandb_entity="senwanxiang",
        work_dir="./results",
        task_name="rps",
        task_type="c",
        task_dim=2,
        save_best_model=False,
    ):
        """
        pretrainer: TCGAPreTrainer
        mode: tuning mode{head:LFT, partial: PFT, or full: FFT}
        """

        self.pretrainer = pretrainer.copy()
        self.scaler = self.pretrainer.scaler

        self.mode = mode
        self.load_decoder = load_decoder

        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.triplet_metric = triplet_metric
        self.triplet_margin = triplet_margin

        self.task_loss_weight = task_loss_weight
        self.task_loss_type = task_loss_type

        self.task_dense_layer = task_dense_layer
        self.task_batch_norms = task_batch_norms
        self.task_class_weight = task_class_weight
        self.batch_correction = batch_correction

        self.entropy_weight = entropy_weight
        self.seed = seed
        fixseed(self.seed)

        self.work_dir = work_dir
        self.with_wandb = with_wandb
        self.wandb_project = wandb_project
        self.wandb_dir = wandb_dir
        self.wandb_entity = wandb_entity
        self.verbose = verbose

        self.task_type = task_type
        self.task_name = task_name
        self.task_dim = task_dim
        self.save_best_model = save_best_model

        self.params = {
            "mode": self.mode,
            "load_decoder": self.load_decoder,
            "device": self.device,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "triplet_margin": self.triplet_margin,
            "triplet_metric": self.triplet_metric,
            "entropy_weight": self.entropy_weight,
            "task_loss_weight": self.task_loss_weight,
            "task_loss_type": self.task_loss_type,
            "task_dense_layer": self.task_dense_layer,
            "task_batch_norms": self.task_batch_norms,
            "task_class_weight": self.task_class_weight,
            "batch_correction": self.batch_correction,
            "work_dir": self.work_dir,
            "with_wandb": self.with_wandb,
            "wandb_project": self.wandb_project,
            "wandb_dir": self.wandb_dir,
            "wandb_entity": self.wandb_entity,
            "verbose": self.verbose,
            "task_dim": self.task_dim,
            "task_type": self.task_type,
            "save_best_model": self.save_best_model,
        }

        __all_loss_type = [
            "ce_loss",
            "focal_loss",
            "mae_loss",
            "dice_loss",
            "dsc_loss",
            "hinge_loss",
        ]
        assert task_loss_type in __all_loss_type, (
            "Unsupported loss type, please input one of %" % __all_loss_type
        )

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime(
            "%Y%m%d_%H%M%S"
        )  # Format: YYYYMMDD_HHMMSS
        run_name = f"Finetune_{self.task_name}_{formatted_time}"  # Example filename

        self.run_name = run_name
        self.save_dir = os.path.join(self.work_dir, run_name)

        if self.with_wandb:
            self.wandb = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                dir=self.wandb_dir,
                name=self.run_name,
                save_code=True,
            )

        ## init model, operimizer, loss ...
        self._reset_state()
        self.best_epoch = 0
        self.feature_name = pretrainer.feature_name

    def _init_model_opt(self):
        """
        pretrainer: TCGAPreTrainer
        """
        fixseed(self.seed)  # Resets the random seed

        model_args = deepcopy(self.pretrainer.saver.inMemorySave["model_args"])
        model_weights = deepcopy(self.pretrainer.saver.inMemorySave["model_state_dict"])

        model_args["task_dim"] = self.task_dim
        model_args["task_type"] = self.task_type  #'f'
        model_args["task_dense_layer"] = self.task_dense_layer
        model_args["task_batch_norms"] = self.task_batch_norms
        model_args["seed"] = self.seed

        ### define finetune model
        model = Compass(**model_args)

        encoder_state = OrderedDict()
        for k, v in model_weights.items():
            if self.load_decoder:
                if self.verbose:
                    print("Load: %s" % k)
                encoder_state[k] = v
            else:
                if not "taskdecoder" in k:
                    if self.verbose:
                        print("Load: %s" % k)
                    encoder_state[k] = v

        ### load Pretrained model
        model.load_state_dict(encoder_state, strict=False)
        model = model.to(self.device)

        if self.mode == "LFT":
            for param in model.inputencoder.parameters():
                param.requires_grad = False
            for param in model.latentprojector.parameters():
                param.requires_grad = False
            plist = [{"params": model.taskdecoder.parameters()}]

        elif self.mode == "PFT":
            for param in model.inputencoder.parameters():
                param.requires_grad = False
            plist = [
                {"params": model.latentprojector.parameters()},  # , 'lr': 5e-3
                {"params": model.taskdecoder.parameters()},  # , 'lr': 1e-3
            ]

        elif self.mode == "FFT":
            # with/wo layer decay
            plist = [
                {"params": model.inputencoder.parameters()},  # , 'lr': 1e-6
                {"params": model.latentprojector.parameters()},  # , 'lr': 1e-5
                {"params": model.taskdecoder.parameters()},  # , 'lr': 1e-3
            ]
        else:
            raise ValueError("Invalid mode type. Use 'PFT','LFT' or 'FFT'. ")

        optimizer = torch.optim.Adam(plist, lr=self.lr, weight_decay=self.weight_decay)
        return model, optimizer

    def _reset_state(self):
        fixseed(self.seed)
        self.model, self.optimizer = self._init_model_opt()
        self.saver = SaveBestModel(save_dir=self.save_dir, save_name="ft_model.pth")

        ssl_loss = TripletLoss(margin=self.triplet_margin, metric=self.triplet_metric)

        if self.task_loss_type == "ce_loss":
            tsk_loss = CEWithNaNLabelsLoss(weights=self.task_class_weight)
        elif self.task_loss_type == "focal_loss":
            tsk_loss = FocalLoss(weights=self.task_class_weight, gamma=2.0)
        elif self.task_loss_type == "mae_loss":
            tsk_loss = MAEWithNaNLabelsLoss()
        elif self.task_loss_type == "hinge_loss":
            tsk_loss = HingeLoss()
        elif self.task_loss_type == "dice_loss":
            tsk_loss = DiceLoss()
        elif self.task_loss_type == "dsc_loss":
            tsk_loss = DSCLoss()

        self.ssl_loss = ssl_loss
        self.tsk_loss = tsk_loss

    def tune(
        self, dfcx_train, dfy_train, min_mcc=0.8, min_f1=0.8, min_roc=0.8, min_prc=0.8
    ):

        self._reset_state()
        if self.with_wandb:
            self.wandb.watch(self.model)

        dfcx_train = self.scaler.transform(dfcx_train)
        train_itrp = ITRPData(dfcx_train, dfy_train)
        train_loader = data.DataLoader(
            train_itrp,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            num_workers=4,
        )  #

        assert train_itrp.y.shape[1] == self.task_dim, "task_dim doesnot matched!"

        # ## init the few-shot taskdecoder based on the representation of the prototypes
        if (self.task_type == "f") & (len(self.task_dense_layer) == 0):
            ## claculate the support set features
            pretrainer = self.pretrainer
            dfcx_train_emb, _ = pretrainer.predict(dfcx_train, batch_size=128)
            proj_feature_names = pretrainer.model.proj_feature_names
            dfcx_train_emb = dfcx_train_emb[proj_feature_names]
            support_features = torch.tensor(dfcx_train_emb.values)
            support_labels = torch.tensor(dfy_train.values)
            self.model.taskdecoder.initialize_parameters(
                support_features, support_labels
            )
            # print(support_features)

        ### training ###
        performance = []
        best_criteria = 1e10
        self.saver(best_criteria, 0, self.model, self.optimizer, self.scaler)

        for epoch in tqdm(range(self.max_epochs), ascii=True):
            train_total_loss, train_ssl_loss, train_tsk_loss = FT_Trainer(
                train_loader,
                self.model,
                self.optimizer,
                self.ssl_loss,
                self.tsk_loss,
                self.device,
                alpha=self.task_loss_weight,
                correction=self.batch_correction,
                entropy_weight=self.entropy_weight,
            )
            train_f1, train_mcc, train_prc, train_roc, train_acc = Evaluator(
                train_loader, self.model, self.device
            )
            performance.append(
                [epoch, train_f1, train_mcc, train_prc, train_roc, train_acc]
            )

            if self.with_wandb:
                self.wandb.log(
                    {
                        "train_f1": train_f1,
                        "train_prc": train_prc,
                        "train_roc": train_roc,
                        "train_acc": train_acc,
                        "train_mcc": train_mcc,
                    }
                )

            if self.verbose:
                print(
                    "Epoch: {}/{} - Train roc: {:.4f}, prc: {:.4f}, f1: {:.4f}, mcc: {:.4f} ".format(
                        epoch + 1,
                        self.max_epochs,
                        train_roc,
                        train_prc,
                        train_f1,
                        train_mcc,
                    )
                )

            # print(train_f1, train_mcc, train_prc, train_roc, train_acc)
            saving_criteria = -train_mcc

            self.saver(saving_criteria, epoch, self.model, self.optimizer, self.scaler)

            if (
                (train_mcc >= min_mcc)
                & (train_f1 >= min_f1)
                & (train_prc >= min_prc)
                & (train_roc >= min_roc)
            ):
                print(
                    "Stopping early at epoch {:2d}. Meet minimal requirements by: f1={:.2f},mcc={:.2f},prc={:.2f}, roc={:.2f}".format(
                        epoch + 1, train_f1, train_mcc, train_prc, train_roc
                    )
                )
                break

        if self.save_best_model:
            self.saver.save()
        else:
            os.system('rm -r "%s"' % self.save_dir)
        self.performance = performance
        self.best_epoch = self.saver.inMemorySave["epoch"]

        ### reload best model weights to model
        self.model.load_state_dict(
            self.saver.inMemorySave["model_state_dict"], strict=True
        )
        self.optimizer.load_state_dict(self.saver.inMemorySave["optimizer_state_dict"])

        return self

    def tune_with_test(
        self,
        dfcx_train,
        dfy_train,
        dfcx_test=None,
        dfy_test=None,
        min_roc=0.8,
        min_prc=0.7,
        min_mcc=0.0,
    ):

        self._reset_state()
        if self.with_wandb:
            self.wandb.watch(self.model)

        ### scaler ####
        dfcx_train = self.scaler.transform(dfcx_train)
        train_itrp = ITRPData(dfcx_train, dfy_train)

        # self.train_itrp = train_itrp
        self.y_scaler = train_itrp.y_scaler
        self.feature_name = train_itrp.feature_name
        train_loader = data.DataLoader(
            train_itrp,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            num_workers=4,
        )  #

        assert self.task_dim == train_itrp.y.shape[1], "task_dim doesnot matched!"

        if dfcx_test is not None:
            dfcx_test = self.scaler.transform(dfcx_test)
            test_itrp = ITRPData(dfcx_test, dfy_test)
            test_loader = data.DataLoader(
                test_itrp,
                batch_size=self.batch_size,
                shuffle=False,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                num_workers=4,
            )
        else:
            test_loader = None

        # ## init the few-shot taskdecoder based on the representation of the prototypes
        if (self.task_type == "f") & (len(self.task_dense_layer) == 0):
            ## claculate the support set features
            pretrainer = self.pretrainer
            dfcx_train_emb, _ = pretrainer.predict(dfcx_train, batch_size=128)
            proj_feature_names = pretrainer.model.proj_feature_names
            dfcx_train_emb = dfcx_train_emb[proj_feature_names]
            support_features = torch.tensor(dfcx_train_emb.values)
            support_labels = torch.tensor(dfy_train.values)
            self.model.taskdecoder.initialize_parameters(
                support_features, support_labels
            )

        ### training ###
        performance = []
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in tqdm(range(self.max_epochs), ascii=True):
            train_total_loss, train_ssl_loss, train_tsk_loss = FT_Trainer(
                train_loader,
                self.model,
                self.optimizer,
                self.ssl_loss,
                self.tsk_loss,
                self.device,
                alpha=self.task_loss_weight,
                correction=self.batch_correction,
                entropy_weight=self.entropy_weight,
            )

            train_f1, train_mcc, train_prc, train_roc, train_acc = Evaluator(
                train_loader, self.model, self.device
            )
            train_avg_metric = np.mean(
                [train_f1, train_mcc, train_prc, train_roc, train_acc]
            )
            saving_criteria = -train_mcc
            # print(train_f1, train_mcc, train_prc, train_roc, train_acc)

            if test_loader is not None:
                test_f1, test_mcc, test_prc, test_roc, test_acc = Evaluator(
                    test_loader, self.model, self.device
                )

                test_total_loss, test_ssl_loss, test_tsk_loss = FT_Tester(
                    test_loader,
                    self.model,
                    self.ssl_loss,
                    self.tsk_loss,
                    self.device,
                    alpha=self.task_loss_weight,
                    correction=0.0,
                )

                test_avg_metric = np.mean(
                    [test_f1, test_mcc, test_prc, test_roc, test_acc]
                )
                saving_criteria = -test_mcc

            else:
                test_total_loss, test_ssl_loss, test_tsk_loss = np.nan, np.nan, np.nan

            ## saving_criteria: lower is better
            ## saving model & Early Stopping based on saving_criteria
            self.saver(saving_criteria, epoch, self.model, self.optimizer, self.scaler)
            if saving_criteria < best_val_loss:
                best_val_loss = saving_criteria
                patience_counter = 0
            else:
                patience_counter += 1

            performance.append(
                [
                    epoch,
                    train_total_loss,
                    train_ssl_loss,
                    train_tsk_loss,
                    test_total_loss,
                    test_ssl_loss,
                    test_tsk_loss,
                ]
            )

            if self.with_wandb:
                self.wandb.log(
                    {
                        "FT_train_loss": train_total_loss,
                        "FT_train_ssl_loss": train_ssl_loss,
                        "FT_train_%s_loss" % self.task_name: train_tsk_loss,
                        "FT_test_loss": test_total_loss,
                        "FT_test_ssl_loss": test_ssl_loss,
                        "FT_test_%s_loss" % self.task_name: test_tsk_loss,
                    }
                )

            if self.verbose:
                print(
                    "Epoch: {}/{} - Train Loss: {:.4f} - Test Loss: {:.4f}".format(
                        epoch + 1, self.max_epochs, train_total_loss, test_total_loss
                    )
                )

            if (
                (patience_counter >= self.patience)
                & (train_roc > min_roc)
                & (train_prc > min_prc)
                & (train_mcc > min_mcc)
            ):
                print(
                    f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {self.patience} consecutive epochs."
                )
                break

        self.performance = performance
        self.best_epoch = self.saver.inMemorySave["epoch"]

        ### reload best model weights to model
        self.model.load_state_dict(
            self.saver.inMemorySave["model_state_dict"], strict=True
        )
        self.optimizer.load_state_dict(self.saver.inMemorySave["optimizer_state_dict"])

        if self.save_best_model:
            self.saver.save()
            ## plot loss locally
            df = pd.DataFrame(
                self.performance,
                columns=[
                    "epochs",
                    "total_loss",
                    "ssl_loss",
                    "%s_loss" % self.task_name,
                    "test_loss",
                    "test_ssl_loss",
                    "test_%s_loss" % self.task_name,
                ],
            ).set_index("epochs")
            # v = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
            fig, ax = plt.subplots(figsize=(7, 5))
            df.plot(ax=ax)
            fig.savefig(
                os.path.join(self.save_dir, "%s_train_loss.png" % self.task_name),
                bbox_inches="tight",
            )
            plt.close()
            df.to_pickle(
                os.path.join(self.save_dir, "%s_train_loss.pkl" % self.task_name)
            )

        else:
            os.system('rm -r "%s"' % self.save_dir)

        return self

    def predict(self, df_tpm, batch_size=512, num_workers=4):
        model = Compass(**self.saver.inMemorySave["model_args"])
        model.load_state_dict(self.saver.inMemorySave["model_state_dict"], strict=False)
        model = model.to(self.device)
        dfe, dfp = Predictor(
            df_tpm,
            model,
            self.scaler,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return dfe, dfp

    def extract(self, df_tpm, batch_size=512, num_workers=4, with_gene_level=False):
        model = Compass(**self.saver.inMemorySave["model_args"])
        model.load_state_dict(self.saver.inMemorySave["model_state_dict"], strict=False)
        model = model.to(self.device)
        dfg, dfgs, dfct = Extractor(
            df_tpm,
            model,
            self.scaler,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
            with_gene_level=True,
        )
        if with_gene_level:
            return dfg, dfgs, dfct
        else:
            return dfgs, dfct

    def project(self, df_tpm, batch_size=512, num_workers=4):
        model = Compass(**self.saver.inMemorySave["model_args"])
        model.load_state_dict(self.saver.inMemorySave["model_state_dict"], strict=False)
        model = model.to(self.device)
        dfg, dfc = Projector(
            df_tpm,
            model,
            self.scaler,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return dfg, dfc

    def plot_embed(self, df_tpm, df_label, label_types, **kwargs):

        ### make prediction & plot on TCGA
        dfe, dfp = self.predict(df_tpm, batch_size=self.batch_size, num_workers=4)
        dfd = dfe.join(df_label)
        label_cols = df_label.columns

        figs = plot_embed_with_label(
            dfd, label_col=label_cols, label_type=label_types, **kwargs
        )
        for fig, name in zip(figs, label_cols):
            fig.savefig(
                os.path.join(self.save_dir, "FT_%s.png" % name),
                bbox_inches="tight",
            )

    def save(self, mfile):
        if self.with_wandb:
            self.wandb._settings = ""
        torch.save(self, mfile)
        print("Saving the model to %s" % mfile)

    def load(self, mfile, **kwargs):
        self = loadcompass(mfile, **kwargs)
        if self.with_wandb:
            self.wandb._settings = ""
        return self

    def close(self):
        if self.with_wandb:
            self.wandb.finish()
            self.wandb._settings = ""
        self.save(os.path.join(self.save_dir, "finetuner.pt"))

    def copy(self):
        return deepcopy(self)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_tuner_epoch(
        self,
        train_X,
        train_y,
        max_epochs=200,
        work_dir="./Paramtune",
        with_wandb=False,
        verbose=False,
        fold=3,
    ):

        param = deepcopy(self.params)
        param["with_wandb"] = with_wandb
        param["epochs"] = max_epochs
        param["work_dir"] = work_dir
        param["verbose"] = verbose

        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
        splits = skf.split(train_y.values, train_y.idxmax(axis=1))

        best_epochs = []
        for i, (train_idx, valid_idx) in enumerate(splits):
            X_train = train_X.iloc[train_idx]
            y_train = train_y.iloc[train_idx]
            X_val = train_X.iloc[valid_idx]
            y_val = train_y.iloc[valid_idx]
            print(
                len(X_train),
                len(X_val),
                y_val.idxmax(axis=1).value_counts().to_dict(),
                y_train.idxmax(axis=1).value_counts().to_dict(),
            )

            tuner = FineTuner(
                self.pretrainer,
                **param,
                task_name="fold_%s" % i,
                task_type="c",
            )

            tuner.tune(
                dfcx_train=X_train, dfy_train=y_train, dfcx_test=X_val, dfy_test=y_val
            )

            best_epoch = tuner.saver.inMemorySave["epoch"]
            best_epochs.append(best_epoch)

        print(np.array(best_epochs))
        best_epochs = int(np.array(best_epochs).mean())

        return best_epochs
