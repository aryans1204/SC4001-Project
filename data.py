import os
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
from PIL import Image
import pytorch_lightning as pl

DATA_DIRS = {
    
    "adience": "C:/Users/teamo/Downloads/data/adience/gender",
    "celeba": "C:/Users/teamo/Downloads/data/celeba",
}

GENDERS = {0: "Female", 1: "Male"}
GENDER_VALUES = {"f": 0, "m": 1}

DATA_TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    ),
}

TRAIN_RATIO = 0.99
BATCH_SIZE = 4


class ImageDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img_label = self.labels[idx]
        img = Image.open(img_path, mode='r')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_label

'''
class AdienceDataset(ImageDataset):
    def __init__(self, root_dir, train=True, train_ratio=0.8, transform=None):
        super().__init__(transform)

        # Iterate through both 'm' and 'f' subdirectories
        for gender_dir in ['m', 'f']:
            gender_path = os.path.join(root_dir, gender_dir)

            for root, _, files in os.walk(gender_path):
                test_start = int(train_ratio * len(files))
                img_files = files[test_start:]

                if train:
                    img_files = files[:test_start]

                for img_file in img_files:
                    img_path = os.path.join(root, img_file)
                    img_label = GENDER_VALUES[gender_dir] # use gender_dir instead of splitting root

                    self.data.append(img_path)
                    self.labels.append(img_label)
'''

class AdienceDataset(ImageDataset):
    def __init__(self, root_dir, partition="train", train_ratio=0.8, transform=None):
        super().__init__(transform)

        self.partition = partition

        # Create a function to partition the files based on the train_ratio
        def split_files(files, ratio):
            split_idx = int(ratio * len(files))
            if partition == "train":
                return files[:split_idx]
            elif partition == "val":
                # For this example, we'll use the rest of the train_ratio for validation. Adjust as needed.
                return files[split_idx:split_idx + int((1 - ratio) * len(files) * 0.5)]
            elif partition == "test":
                return files[split_idx + int((1 - ratio) * len(files) * 0.5):]
            else:
                raise ValueError("Invalid partition")

        # Iterate through both 'm' and 'f' subdirectories
        for gender_dir in ['m', 'f']:
            gender_path = os.path.join(root_dir, gender_dir)

            for root, _, files in os.walk(gender_path):
                img_files = split_files(files, train_ratio)

                for img_file in img_files:
                    img_path = os.path.join(root, img_file)
                    img_label = GENDER_VALUES[gender_dir]  # use gender_dir instead of splitting root

                    self.data.append(img_path)
                    self.labels.append(img_label)

adience_train_dataset = AdienceDataset(
    DATA_DIRS["adience"],
    partition="train",
    transform=DATA_TRANSFORMS["train"],
)
adience_test_dataset = AdienceDataset(
    DATA_DIRS["adience"],
    partition="test",
    transform=DATA_TRANSFORMS["test"],
)

adience_val_dataset = AdienceDataset(
    DATA_DIRS["adience"],
    partition="val",
    transform=DATA_TRANSFORMS["val"],
)

# Dataloaders
# Adience
adience_train_dataloader = DataLoader(
    adience_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
adience_test_dataloader = DataLoader(
    adience_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

class AdienceMod(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train_tr, val_tr, test_tr):
        super(AdienceMod, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_tr = train_tr
        self.val_tr = val_tr
        self.test_tr = test_tr
    
    def setup(self, stage=None):
        adience_train_dataset = AdienceDataset(
            self.root_dir, partition="train", transform=self.train_tr
        )
        adience_test_dataset = AdienceDataset(
            self.root_dir, partition="test", transform=self.test_tr
        )
        adience_val_dataset = AdienceDataset(
            self.root_dir, partition="val", transform=self.val_tr
        )
        self.train_ds = adience_train_dataset
        self.val_ds = adience_val_dataset
        self.test_ds = adience_test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

class CelebADataset(ImageDataset):
    def __init__(self, root_dir, partition="train", transform=None):
        super().__init__(transform)

        img_partition_dict = {}

        with open(os.path.join(root_dir, "list_eval_partition.txt")) as eval_file:
            for line in eval_file:
                tokens = line.strip().split()
                img_filename = tokens[0]
                img_partition = ["train", "val", "test"][int(tokens[1])]
                img_partition_dict[img_filename] = img_partition

        with open(os.path.join(root_dir, "list_attr_celeba.txt")) as attr_file:
            next(attr_file)
            next(attr_file)

            for line in attr_file:
                tokens = line.strip().split()
                img_filename = tokens[0]

                if img_partition_dict[img_filename] != partition:
                    continue

                img_path = os.path.join(root_dir, "img_align_celeba/img_align_celeba", tokens[0])
                img_label = int(tokens[21])

                self.data.append(img_path)
                self.labels.append((img_label + 1) // 2)


# Datasets
# Adience

# CelebA
celeba_train_dataset = CelebADataset(
    DATA_DIRS["celeba"], partition="train", transform=DATA_TRANSFORMS["train"]
)
celeba_test_dataset = CelebADataset(
    DATA_DIRS["celeba"], partition="test", transform=DATA_TRANSFORMS["test"]
)
celeba_val_dataset = CelebADataset(
    DATA_DIRS["celeba"], partition="val", transform=DATA_TRANSFORMS["val"]
)



# CelebA
celeba_train_dataloader = DataLoader(
    celeba_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
celeba_test_dataloader = DataLoader(
    celeba_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
celeba_val_dataloader = DataLoader(
    celeba_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

class CelebAMod(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train_tr, val_tr, test_tr):
        super(CelebAMod, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_tr = train_tr
        self.val_tr = val_tr
        self.test_tr = test_tr

    def setup(self, stage=None):
        celeba_train_dataset = CelebADataset(
            self.root_dir, partition="train", transform=self.train_tr
        )
        celeba_test_dataset = CelebADataset(
            self.root_dir, partition="test", transform=self.val_tr
        )
        celeba_val_dataset = CelebADataset(
            self.root_dir, partition="val", transform=self.test_tr
        )
        self.train_ds = celeba_train_dataset
        self.val_ds = celeba_val_dataset
        self.test_ds = celeba_test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
    


'''
print(len(adience_train_dataset))
print(len(adience_test_dataset))
print(len(adience_val_dataset))

sample = adience_train_dataset[0]  # Get the first sample
image, label = sample
print(f'Image shape: {image.shape}, Label: {label}')
'''

celebds = CelebAMod(DATA_DIRS["celeba"], BATCH_SIZE, DATA_TRANSFORMS["train"], DATA_TRANSFORMS["val"], DATA_TRANSFORMS["test"])
adienceds = AdienceMod(DATA_DIRS["adience"], BATCH_SIZE, DATA_TRANSFORMS["train"], DATA_TRANSFORMS["val"], DATA_TRANSFORMS["test"])