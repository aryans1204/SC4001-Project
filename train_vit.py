
from data import celebds
from data import adienceds

import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.trainer import Trainer
import torch.nn.functional as F

import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

from IPython.core.display import display, HTML
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class LitModel(pl.LightningModule):
    def __init__(
        self,
        n_classes = 2, #created a default value
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
        self.loss = nn.BCELoss()
        '''self.loss = nn.CrossEntropyLoss()'''
        # Used to log images preds to wandb
        self.opt_params = {
            "lr": 0.001,
        }
        self.validation_step_outputs = []
        self.train_accuracies = []
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.model(x)
        x = nn.Sigmoid()(x)
        return torch.tensor(x >= 0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.forward(x)
        loss = self.loss(y_hat_logits.float(), torch.unsqueeze(y, -1).float().to('cuda:0'))
        loss.requires_grad = True
        #train_acc = self.train_accuracy(y_hat_logits.float(), y.type(torch.FloatTensor).view(y.size(0), 1).to('cuda:0'))
        #self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_loss", loss)
        #self.train_accuracies.append(train_acc)
        self.train_losses.append(loss)
        return {"loss": loss, "progress_bar": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.forward(x)
        #self.val_accuracy(y_hat_logits, y)
        val_loss = self.loss(y_hat_logits.float(), torch.unsqueeze(y, -1).float().to('cuda:0'))
        self.val_losses.append(val_loss)
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
        #self.test_accuracy(y_hat_logits.floa, y.type(torch.FloatTensor))
        test_loss = self.loss(y_hat_logits, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_params)
        return optimizer

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
    
class ViTModel(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels.to('cuda:0'))
        acc = (preds.argmax(dim=-1) == labels.to('cuda:0')).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

def train_vit(**kwargs):
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=8,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join("", "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = ViTModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = ViTModel(**kwargs)
        trainer.fit(model, adienceds)
        # Load best checkpoint after training
        model = ViTModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    '''val_result = trainer.test(model, celebds, verbose=False)
    test_result = trainer.test(model, celebds, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}'''

    return model

model = train_vit(
    model_kwargs={
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "patch_size": 4,
        "num_channels": 3,
        "num_patches": 64*64,
        "num_classes": 2,
        "dropout": 0.2,
    },
    lr=3e-4,
)
model = LitModel(1)

ckpt = ModelCheckpoint(monitor="val_loss_epoch")

trainer = Trainer(max_epochs=2, callbacks=[ckpt])

trainer.fit(model, adienceds)

count = 0
pos_count = 0
neg_count = 0
with torch.no_grad():
    for batch in adienceds.test_dataloader():
        inputs, labels = batch
        outputs = model(inputs)
        inputs_cpu = inputs.cpu().numpy()
        predicted_labels = (outputs > 0.5).int()
        labels_cpu = labels.cpu().numpy()
        predicted_labels_cpu = predicted_labels.cpu().numpy()
        if count < 5:
            plt.imshow(np.transpose(inputs_cpu[0], (1, 2, 0)))
            plt.title(f"Actual: {labels_cpu[0]}, Predicted: {predicted_labels_cpu[0]}")
            plt.show()
        if int(labels_cpu[0]) == int(predicted_labels_cpu[0]):
            pos_count += 1
        else:
            neg_count += 1
        count+=1
        
print(f"The number of positive matches is {pos_count}")
print(f"The number of negative matches is {neg_count}")



        
    

