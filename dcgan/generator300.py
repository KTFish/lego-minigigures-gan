# DCGAN Generator for 300x300 pixel images (around the original resolution of images)
import torch
from torch import nn
import config as c
import utils

class Generator300(nn.Module):
    def __init__(self, z_dim: int=c.Z_DIM, im_chan: int=c.IMAGE_CHANNELS, hc: int=c.HIDDEN_CHANNELS_GEN):
        """Generator Model.

        Args:
            z_dim (int, optional): Dimension of the noise vector. Defaults to c.Z_DIM.
            im_chan (int, optional): Number of channels in the image (here 3, because it is a RGB image). Defaults to c.IMAGE_CHANNELS.
            hc (int, optional): Hidden Channels, a short name is used for better code readability. Defaults to c.HIDDEN_CHANNELS_GEN.
        """
        super(Generator300, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.block(z_dim, hc * 32, 3, 2), # 1x1 --> 3x3
            self.block(hc * 32, hc * 16, 4, 1), # 3x3 --> 6x6
            self.block(hc * 16, hc * 8, 3, 2), # 6x6 --> 13x13
            self.block(hc * 8, hc * 4, 4, 2), # 13x13 --> 28x28
            self.block(hc * 4, hc * 2, 2, 2), # 28x28 --> 56x56
            self.block(hc * 2, hc, 2,2), # 56x56 --> 112x122
            self.block(hc, im_chan, 2,2, final_layer=True),  # 112x112 --> 224x224
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



def calculate_transpose_settings(input_size, output_size, kernel_size):
    stride = (output_size - 1) // (input_size - 1)
    padding = (kernel_size - 1) - ((input_size + (stride - 1) * (input_size - 1)) - output_size)
    return stride, padding


def test_generator_output_shape() -> None:
    """Function tests if the output shape matches the desired one."""
    gen = Generator300().to(c.DEVICE)
    noise = utils.sample_noise(c.BATCH_SIZE, c.Z_DIM)
    output = gen(noise)

    height, width = c.RESIZE_TO_SHAPE
    assert output.shape == torch.Size([c.BATCH_SIZE, c.IMAGE_CHANNELS, height, width])


if __name__ == "__main__":
   test_generator_output_shape()