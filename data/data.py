import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
from PIL import Image

DATA_DIRS = {
    "adience": "adience/gender",
    "celeba": "celeba",
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
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_label


class AdienceDataset(ImageDataset):
    def __init__(self, root_dir, train=True, train_ratio=0.8, transform=None):
        super().__init__(transform)

        for root, _dirs, files in os.walk(root_dir):
            test_start = int(train_ratio * len(files))
            img_files = files[test_start:]

            if train:
                img_files = files[:test_start]

            for img_file in img_files:
                img_path = os.path.join(root, img_file)
                img_label = GENDER_VALUES[os.path.split(root)[-1]]

                self.data.append(img_path)
                self.labels.append(img_label)


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

                img_path = os.path.join(root_dir, "img_align_celeba", tokens[0])
                img_label = int(tokens[21])

                self.data.append(img_path)
                self.labels.append((img_label + 1) // 2)


# Datasets
# Adience
adience_train_dataset = AdienceDataset(
    DATA_DIRS["adience"],
    train=True,
    train_ratio=TRAIN_RATIO,
    transform=DATA_TRANSFORMS["train"],
)
adience_test_dataset = AdienceDataset(
    DATA_DIRS["adience"],
    train=False,
    train_ratio=TRAIN_RATIO,
    transform=DATA_TRANSFORMS["test"],
)

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

# Dataloaders
# Adience
adience_train_dataloader = DataLoader(
    adience_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
adience_test_dataloader = DataLoader(
    adience_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
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
