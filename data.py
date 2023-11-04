import os
import random
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

SEED = 69

DATA_DIRS = {
    "utkface": "data/utkface",
    "celeba": "C:/Users/teamo/Downloads/data/celeba",
}

GENDERS = {0: "Female", 1: "Male"}

IMAGE_SIZE = 224
DATA_TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
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
        img = Image.open(img_path, mode="r")

        if self.transform is not None:
            img = self.transform(img)

        return img, img_label


class UTKFaceDataset(ImageDataset):
    def __init__(
        self, root_dir, partition="train", train_ratio=TRAIN_RATIO, transform=None
    ):
        super().__init__(transform)

        random.seed(SEED)
        val_ratio = (1 - train_ratio) / 2

        for root, _, files in os.walk(root_dir):
            random.shuffle(files)
            file_count = len(files)

            img_files = files[: int(file_count * train_ratio)]
            if partition == "val":
                img_files = files[
                    int(file_count * train_ratio) : int(
                        file_count * (train_ratio + val_ratio)
                    )
                ]
            elif partition == "test":
                img_files = files[int(file_count * (train_ratio + val_ratio)) :]

            for img_file in img_files:
                img_path = os.path.join(root, img_file)

                # [age]_[gender]_[race]_[date&time].jpg
                # 0 - male, 1 - female
                img_label = abs(int(img_file.split("_")[1]) - 1)

                self.data.append(img_path)
                self.labels.append(img_label)


class UTKFaceMod(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train_tr, val_tr, test_tr):
        super(UTKFaceMod, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_tr = train_tr
        self.val_tr = val_tr
        self.test_tr = test_tr

    def setup(self, stage=None):
        self.train_ds = UTKFaceDataset(
            self.root_dir, partition="train", transform=self.train_tr
        )
        self.val_ds = UTKFaceDataset(
            self.root_dir, partition="val", transform=self.test_tr
        )
        self.test_ds = UTKFaceDataset(
            self.root_dir, partition="test", transform=self.val_tr
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )


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

                img_path = os.path.join(
                    root_dir, "img_align_celeba/img_align_celeba", tokens[0]
                )
                img_label = int(tokens[21])

                self.data.append(img_path)
                self.labels.append((img_label + 1) // 2)


class CelebAMod(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train_tr, val_tr, test_tr):
        super(CelebAMod, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_tr = train_tr
        self.val_tr = val_tr
        self.test_tr = test_tr

    def setup(self, stage=None):
        self.train_ds = CelebADataset(
            self.root_dir, partition="train", transform=self.train_tr
        )
        self.val_ds = CelebADataset(
            self.root_dir, partition="val", transform=self.val_tr
        )
        self.test_ds = CelebADataset(
            self.root_dir, partition="test", transform=self.test_tr
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )


"""
print(len(adience_train_dataset))
print(len(adience_test_dataset))
print(len(adience_val_dataset))

sample = adience_train_dataset[0]  # Get the first sample
image, label = sample
print(f"Image shape: {image.shape}, Label: {label}")
"""

celebds = CelebAMod(
    DATA_DIRS["celeba"],
    BATCH_SIZE,
    DATA_TRANSFORMS["train"],
    DATA_TRANSFORMS["val"],
    DATA_TRANSFORMS["test"],
)
utkfaceds = UTKFaceMod(
    DATA_DIRS["utkface"],
    BATCH_SIZE,
    DATA_TRANSFORMS["train"],
    DATA_TRANSFORMS["val"],
    DATA_TRANSFORMS["test"],
)
