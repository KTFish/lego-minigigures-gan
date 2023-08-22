import torch
import config
import utils
from torch import nn

# Strided Convolutions
# Paper: "Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)."
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html


class GenBlock(nn.Module):
    """Block of generator network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        final_layer: bool = False,
    ) -> None:
        super().__init__()
        # Paper: "Use ReLU activation in generator for all layers except for the output, which uses Tanh."
        # Paper: "Directly applying batchnorm to all layer however, resulted in sample oscillation and model instability.
        # This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer."

        # ConvTranspose shape formula: n = stride * (input_shape - 1) + kernel_size - 2 * padding
        if final_layer:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.Tanh(),  # Output [-1, 1]
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int = 100,
        out_channels: int = 1,
        hidden_units: int = config.HIDDEN_UNITS,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            GenBlock(z_dim, hidden_units),
            GenBlock(hidden_units, hidden_units // 2),
            GenBlock(hidden_units // 2, out_channels, 10, 2, True),
            nn.Tanh()
        )

    def unsqueeze_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise) -> torch.Tensor:
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


def test_generator() -> None:
    gen = Generator()
    n_samples, z_dim = 32, 100

    assert gen.z_dim == z_dim

    # Test shape of noise vector
    noise = utils.sample_noise(n_samples, z_dim)
    assert noise.shape == torch.Size([n_samples, z_dim])

    # Test unsqueezed noise shape
    unsqueezed_noise = gen.unsqueeze_noise(noise)
    assert unsqueezed_noise.shape == torch.Size([n_samples, z_dim, 1, 1])

    # Test if output of generator is the same as the desired shape
    real_images_batch = torch.rand(n_samples, 1, 28, 28)
    fake_images_batch = gen(noise)
    assert (
        real_images_batch.shape == fake_images_batch.shape
    ), f"Wrong generator output shape. Real Image shape {real_images_batch.shape}, Fake Image shape: {fake_images_batch.shape}."


def test_gen_block():
    n_samples, z_dim, hidden_units = 32, 100, 1024
    noise = utils.sample_noise(n_samples, z_dim)
    x = Generator().unsqueeze_noise(noise)

    # Test first Generator Block
    x = GenBlock(z_dim, hidden_units)(x)
    assert x.shape == torch.Size([n_samples, 1024, 4, 4])


if __name__ == "__main__":
    print(f"Running Generator tests...")
    test_gen_block()
    test_generator()
    print("All tests passed successfully!")
