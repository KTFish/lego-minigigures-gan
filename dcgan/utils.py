import os
import torch
import config
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Tuple
from torchvision.utils import make_grid, save_image

def save_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), img_type:str='real', epoch:int=0,step=0):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    save_image(image_grid, f"dcgan\images\{img_type}\epoch-{epoch}-step-{step}.png")


def print_training_progress(
    epoch: int, step: int, gen_loss: float, disc_loss: float
) -> None:
    """Prints training progress given data.

    Args:
        epoch (int): _description_
        gen_loss (float): _description_
        disc_loss (float): _description_
    """
    print(
        f"Epoch: {epoch} | Step: {step} | Generator Loss: {gen_loss:.3f} | Discriminator Loss: {disc_loss:.3f}"
    )


def sample_noise(n_samples: int, z_dim: int, device: str = "cpu") -> torch.Tensor:
    """Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the uniform distribution.

    Args:
        n_samples (int): Number of samples that should be generated.
        z_dim (int): Lenght of the Z vector.
        device (str, optional): Device of the tensor. Defaults to "cpu".

    Returns:
        torch.Tensor: Noise vector of dimensions (n_samples, z_dim)
    """
    return torch.randn(n_samples, z_dim, device=device)


def weights_init(m):
    """Weights initialization to the normal distribution with mean 0 and standard deviation 0.02 like in dcgan paper.
    Args:
         m (nn.Module): A PyTorch module for which weights will be initialized.

    Example usage:
        gen = gen.apply(weights_init)  # Initializing weights in generator
        disc = disc.apply(weights_init)  # Initializing weights in discriminator

    Note:
        This function is typically used with the `.apply()` method of a PyTorch module to apply weight
        initialization to all relevant layers within the module.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def save_model_checkpoint(
    model: torch.nn.Module, name: str, path: str = "./models/dcgan/"
) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    full_path = f"{path}/{name}.pth"
    print(f"Saving {name} to {full_path}.")
    torch.save(model.state_dict(), full_path)


def print_training_info() -> None:
    print(
        f"DCGAN starts training on {config.DEVICE} for {config.EPOCHS} epochs. Other parameters of training are:\n- Batch Size: {config.BATCH_SIZE}\n- Learning Rate: {config.LEARNING_RATE}\n- Betas: {config.BETAS}\n- Hidden Units: {config.HIDDEN_UNITS}\n- Z_DIM: {config.Z_DIM}"
    )


def setup_generated_image_folders() -> None:
    """
    Set up folders for generated images.

    This function creates the necessary folder structure to organize generated
    images into 'fake' and 'real' subfolders under the './images' directory.
    """
    path = "./dcgan/images"
    fake = "./dcgan/images/fake"
    real = "./dcgan/images/real"
    # Create folders for generated images
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(fake):
        os.makedirs(fake)
    if not os.path.exists(real):
        os.makedirs(real)