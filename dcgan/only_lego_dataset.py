import os
import torch
import config
import pathlib
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class OnlyLegoDataset(Dataset):
    """Dataset containing only images of lego minifigures, mannualy selected from the scrapped data.
    It contains 11459 images originally in different sizes, but all sqares.
    """

    def __init__(self, root: str, transform=None) -> None:
        super().__init__()  # ?

        self.root = root
        self.paths = list(pathlib.Path(root).glob("*.png"))
        self.transform = transform

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)

        # Transform if necessary
        if self.transform:
            return self.transform(img)
        else:
            return img


def get_datasets() -> Tuple[OnlyLegoDataset]:
    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.RESIZE_TO_SHAPE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(config.IMAGE_CHANNELS)], [0.5 for _ in range(config.IMAGE_CHANNELS)])
        ]
    )

    dataset = OnlyLegoDataset(
        root=os.path.join('.', 'only_lego_dataset_preprocessed'), transform=train_transforms
    )

    return dataset


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
