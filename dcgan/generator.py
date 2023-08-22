import torch
from torch import nn
import config as c

class Generator(nn.Module):
    def __init__(self, z_dim: int=c.Z_DIM, im_chan: int=c.IMAGE_CHANNELS, hc: int=c.HIDDEN_CHANNELS):
        """Generator Model.

        Args:
            z_dim (int, optional): Dimension of the noise vector. Defaults to c.Z_DIM.
            im_chan (int, optional): Number of channels in the image (here 3, because it is a RGB image). Defaults to c.IMAGE_CHANNELS.
            hc (int, optional): Hidden Channels, a short name is used for better code readability. Defaults to c.HIDDEN_CHANNELS.
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.block(z_dim, hc * 4),
            self.block(hc * 4, hc * 2, kernel_size=4, stride=1),
            self.block(hc * 2, hc),
            self.block(hc, im_chan, kernel_size=4, final_layer=True),
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        stride: int=2,
        final_layer: bool=False,
    )-> nn.Sequential:
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.Tanh())

    def unsqueeze_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Takes a noise tensor as input and unsqueezes it to match the dimensions required by the generator's input layer.

        Args:
            noise (torch.Tensor): Noise tensor with dimensions (n_samples, z_dim) that will be unsqueezed.

        Returns:
            torch.Tensor: The unsqueezed noise tensor with dimensions suitable for the generator's input (width = 1, height = 1 and channels = z_dim).
        """
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Generator Forward Pass Function.
        When provided with a noise tensor, this function produces generated images as its output.

        Args:
            noise (torch.Tensor): Noise tensor with dimensions (n_samples, z_dim).

        Returns:
            torch.Tensor: Generated images.
        """
        x = self.unsqueeze_noise(noise)
        return self.gen(x)
