
from data import celebds

import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.trainer import Trainer

import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

from IPython.core.display import display, HTML
from pytorch_lightning.callbacks import ModelCheckpoint


class LitModel(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        download_pretrained=True,
        wandb_logger=None,
        **kwargs,
    ):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, n_classes)
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')
        self.loss = nn.CrossEntropyLoss()
        # Used to log images preds to wandb
        self.opt_params = {
            "lr": 0.001,
        }
        self.validation_step_outputs = []

    def forward(self, x):
        x = self.model(x)
        x = torch.argmax(x, dim=1)
        x = x.view(x.size(0), 1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.forward(x)
        loss = self.loss(y_hat_logits.float(), y.float().view(y.size(0), 1).to('cuda:0'))
        loss.requires_grad = True
        train_acc = self.train_accuracy(y_hat_logits.float(), y.type(torch.FloatTensor).view(y.size(0), 1).to('cuda:0'))
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_loss", loss)
        return {"loss": loss, "progress_bar": {"train_acc": train_acc}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.forward(x)
        #self.val_accuracy(y_hat_logits, y)
        val_loss = self.loss(y_hat_logits.float(), y.float().view(y.size(0), 1).to('cuda:0'))
        val_loss.requires_grad = True
        self.validation_step_outputs.append(val_loss)
        return {"val_loss": val_loss, "out_logits": y_hat_logits}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log("val_acc_epoch", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_loss_epoch", avg_val_loss, on_epoch=True, prog_bar=True)
        return {
            "val_loss": avg_val_loss,
        }
        

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.forward(x)
        self.test_accuracy(y_hat_logits.float(), y.type(torch.FloatTensor))
        test_loss = self.loss(y_hat_logits.float(), y.type(torch.FloatTensor))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_params)
        return optimizer
    

model = LitModel(2)
ckpt = ModelCheckpoint()

trainer = Trainer(max_epochs=6, callbacks=[ckpt])

trainer.fit(model, celebds)



        
    

