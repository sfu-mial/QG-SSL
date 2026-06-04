import albumentations as A
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class LabeledSegmentationDataset(Dataset):
    """
    Dataset for labeled segmentation masks.
    """

    def __init__(
        self,
        csv_path: str,
        image_size: int,
        is_train: bool = True,
        strong_aug: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.is_train = is_train
        self.strong_aug = strong_aug

        if "img_path" in self.df.columns:
            self.df = self.df.rename(columns={"img_path": "image_path"})
        if "seg_path" in self.df.columns:
            self.df = self.df.rename(columns={"seg_path": "mask_path"})

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if is_train:
            if strong_aug:
                self.transform = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Rotate(limit=30, p=0.7),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.3, p=0.7
                        ),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                        A.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1,
                            p=0.5,
                        ),
                        A.Normalize(mean=imagenet_mean, std=imagenet_std),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Resize(image_size, image_size),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Rotate(limit=20, p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.Normalize(mean=imagenet_mean, std=imagenet_std),
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].unsqueeze(0).float()


class UnlabeledDataset(Dataset):
    """
    Dataset for unlabeled images.
    """

    def __init__(
        self, csv_path: str, image_size: int, return_both_augs: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.return_both_augs = return_both_augs

        if "image_path" in self.df.columns:
            self.image_col = "image_path"
        elif "img_path" in self.df.columns:
            self.image_col = "img_path"
        else:
            self.image_col = self.df.columns[0]

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.weak_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=imagenet_mean, std=imagenet_std),
                ToTensorV2(),
            ]
        )

        self.strong_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.7
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5,
                ),
                A.Normalize(mean=imagenet_mean, std=imagenet_std),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row[self.image_col])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.return_both_augs:
            weak = self.weak_transform(image=image)["image"]
            strong = self.strong_transform(image=image)["image"]
            return weak, strong
        else:
            return self.weak_transform(image=image)["image"]
