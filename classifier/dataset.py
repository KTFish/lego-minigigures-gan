import torch
import torchvision
import PIL
import os
import pathlib
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict


# TODO: Change the dataset so that it has the aroppriate structure

# TODO: Write custom dataset: https://www.learnpytorch.io/04_pytorch_custom_datasets/#52-create-a-custom-dataset-to-replicate-imagefolder
# TODO: Write transforms, resize the image to a smaller size, do some augmentation
# TODO: Creata dataloaders


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class LegoBinaryDataset(Dataset):
    """Lego vs. Not Lego Dataset."""

    def __init__(self, root: str, transform=None) -> None:
        super().__init__()  # ?

        self.root = root
        # self.paths = list(pathlib.Path(r".\preprocessed_dataset\train").glob("*/*.png"))
        self.paths = list(pathlib.Path(root).glob("*/*.png"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(root)

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
        class_name = self.paths[
            index
        ].parent.name  # expects path in data_folder/class_name/image.png
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return data, label (X, y)


def get_datasets() -> Tuple[LegoBinaryDataset, LegoBinaryDataset]:
    train_transforms = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

    train_dataset = LegoBinaryDataset(
        root=r"preprocessed_dataset\train", transform=train_transforms
    )

    test_dataset = LegoBinaryDataset(
        root=r"preprocessed_dataset\test", transform=test_transforms
    )
    return train_dataset, test_dataset


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
