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
import numpy as np

from IPython.core.display import display, HTML
from pytorch_lightning.callbacks import ModelCheckpoint


class LitModel(pl.LightningModule):
    def __init__(
        self,
        n_classes = 1, #created a default value
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
        self.avg_training_accuracies = []

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
        #x = nn.Sigmoid()(x)
        #return torch.tensor(x >= 0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.forward(x)
        loss = self.loss(y_hat_logits.float(), torch.unsqueeze(y, -1).float().to('cuda:0'))
        train_acc = self.train_accuracy(y_hat_logits.float(), y.type(torch.FloatTensor).view(y.size(0), 1).to('cuda:0'))
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_loss", loss)
        self.train_accuracies.append(train_acc)
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
    
'''
def visualize_samples(dataloader, num_samples=5):
    batch = next(iter(dataloader))
    images, labels = batch
    
    for idx in range(min(num_samples, len(images))):
        plt.imshow(np.transpose(images[idx].numpy(), (1, 2, 0)))  # Convert CxHxW to HxWxC for visualization
        plt.title(f'Label: {labels[idx]}')
        plt.show()
'''
#model = LitModel(1)

adienceds.setup()
ckpt = ModelCheckpoint()
model = LitModel.load_from_checkpoint('epoch=7-step=325544.ckpt')
trainer = Trainer(max_epochs=1, callbacks=[ckpt])
trainer.fit(model, train_dataloaders=adienceds.train_dataloader(), val_dataloaders=adienceds.val_dataloader())

diff_length = len(model.train_losses) - len(model.val_losses)

# # Extend val_losses with NaN values to match the length


train_losses_cpu = [loss_item.detach().cpu().numpy() for loss_item in model.train_losses]
train_accuracies_cpu = [loss_item.detach().cpu().numpy() for loss_item in model.train_accuracies]
val_losses_cpu = [loss_item.detach().cpu().numpy() for loss_item in model.val_losses]

val_losses_cpu.extend([np.nan] * diff_length)

df = pd.DataFrame({'Training Losses': train_losses_cpu,
                  'Training Accuracies': train_accuracies_cpu,
                  'Validation Losses': val_losses_cpu})

df.to_csv('losses.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Plot 'Training Losses' on the first subplot
axes[0].plot(df['Training Losses'], label='Training Losses', color='blue')
axes[0].set_title('Training Losses')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Plot 'Validation Losses' on the second subplot
axes[1].plot(df['Validation Losses'], label='Validation Losses', color='red')
axes[1].set_title('Validation Losses')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

# Show the plot
plt.savefig("losses_plot.png")
plt.tight_layout()
plt.show()



# model.eval()

# predictions = []
# actuals = []

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
  # Assuming images are in [C, H, W] format

# num_images_to_show = 5

# for i in range(num_images_to_show):
#     plt.imshow(np.transpose(adienceds.test_ds[i][0].numpy(), (1, 2, 0)))  # Assuming images are in [C, H, W] format
#     plt.title(f"Actual: {actuals[i]}, Predicted: {predictions[i]}")
#     plt.show()



        
    

