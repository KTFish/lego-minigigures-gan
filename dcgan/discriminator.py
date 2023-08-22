import torch
from torch import nn
import config


class DiscBlock(nn.Module):
    """Block of Discriminator Model for the MNIST Dataset."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        input_layer: bool = False,
        final_layer: bool = False,
    ) -> None:
        super().__init__()
        if input_layer:  # No use of batchnorm
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.LeakyReLU(config.LEAKY_RELU_SLOPE),
            )
        elif final_layer:  # Use additionally Flatten and Sigmoid
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                # ! On Coursera they do not use activation in the last layer. Why?
                # nn.LeakyReLU(config.LEAKY_RELU_SLOPE),
                # nn.Flatten(),  # Flatten the output of last conv layer
                nn.Sigmoid(),  # Binary Classification between real and fake
            )

        else:  # Use batchnorm
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(config.LEAKY_RELU_SLOPE),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, hidden_units: int = config.HIDDEN_UNITS
    ) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            DiscBlock(
                in_channels, hidden_units, kernel_size=5, stride=2, input_layer=True
            ),
            DiscBlock(hidden_units, hidden_units * 2, 5, 2),
            DiscBlock(hidden_units * 2, hidden_units * 4, 5, 2),
            DiscBlock(hidden_units * 2, out_channels, 4, 2, False, final_layer=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.disc(x)
        return x


def test_discriminator() -> None:
    disc = Discriminator(in_channels=3, out_channels=1, hidden_units=512)
    fake_image = torch.randn(config.BATCH_SIZE, 3, 28, 28)
    pred = disc(fake_image).squeeze()

    # Test shape of discriminator output
    assert pred.shape == torch.Size(
        [config.BATCH_SIZE]
    ), f"{pred.shape} != {torch.Size([config.BATCH_SIZE])}"

    # Test the range of discriminator output (it should be between 0 and 1 due to Sigmoid function)
    pred = torch.sigmoid(pred)
    assert (
        pred.max() <= 1
    ), f"Max probability shoud be less or equal to 1, not {pred.max()}."
    assert (
        pred.min() >= 0
    ), f"Min probability shoud be less or equal to 1, not {pred.min()}."


if __name__ == "__main__":
    print("Running discriminator tests...")
    test_discriminator()
    print("All tests passed successfully!")
