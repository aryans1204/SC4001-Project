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

import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import numpy as np

from IPython.core.display import display, HTML
from pytorch_lightning.callbacks import ModelCheckpoint


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
        self.train_accuracies.append(train_acc)
        self.train_losses.append(loss)
        return {"loss": loss, "progress_bar": {"train_acc": train_acc}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.forward(x)
        #self.val_accuracy(y_hat_logits, y)
        val_loss = self.loss(y_hat_logits.float(), y.type(torch.FloatTensor).view(y.size(0), 1).to('cuda:0'))
        self.val_losses.append(val_loss)
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
    
'''
def visualize_samples(dataloader, num_samples=5):
    batch = next(iter(dataloader))
    images, labels = batch
    
    for idx in range(min(num_samples, len(images))):
        plt.imshow(np.transpose(images[idx].numpy(), (1, 2, 0)))  # Convert CxHxW to HxWxC for visualization
        plt.title(f'Label: {labels[idx]}')
        plt.show()
'''
model = LitModel(2)

adienceds.setup()
ckpt = ModelCheckpoint()
#model = LitModel.load_from_checkpoint('epoch=5-step=244158.ckpt')
trainer = Trainer(max_epochs=10, callbacks=[ckpt])
trainer.fit(model, train_dataloaders=adienceds.train_dataloader(), val_dataloaders=adienceds.val_dataloader())

print(len(model.train_losses))
print(len(model.val_losses))
print(len(model.train_accuracies))

diff_length = len(model.train_losses) - len(model.val_losses)

# Extend val_losses with NaN values to match the length
model.val_losses.extend([np.nan] * diff_length)

df = pd.DataFrame({'Training Losses': model.train_losses,
                   'Training Acccuracies': model.train_accuracies,
                   'Validation Losses': model.val_losses})
df.to_csv('losses.csv', index=False)

plt.figure(figsize=(10, 5))
plt.plot(model.train_losses, label='Training Losses', color='blue')
plt.plot(model.val_losses, label='Validation Losses', color='red')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("losses_plot.png")
plt.show()

model.eval()

predictions = []
actuals = []

with torch.no_grad():
    for batch in adienceds.test_dataloader():
        inputs, labels = batch
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

num_images_to_show = 5

for i in range(num_images_to_show):
    plt.imshow(np.transpose(adienceds.test_ds[i][0].numpy(), (1, 2, 0)))  # Assuming images are in [C, H, W] format
    plt.title(f"Actual: {actuals[i]}, Predicted: {predictions[i]}")
    plt.show()



        
    

