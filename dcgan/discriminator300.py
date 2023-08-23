import torch
from torch import nn
import config as c

class Discriminator300(nn.Module):
    def __init__(self, im_chan: int=c.IMAGE_CHANNELS, hc: int=c.HIDDEN_CHANNELS_DISC):
        """Discriminator Model.

        Args:
            im_chan (int, optional): Channels of the input image. Defaults to c.IMAGE_CHANNELS.
            hc (int, optional): Hidden Channels of the model. Defaults to c.HIDDEN_CHANNELS_DISC.
        """
        super(Discriminator300, self).__init__()
        self.disc = nn.Sequential(
            self.block(im_chan, hc, 2, 2),      # 224x224 --> 112x112
            self.block(hc, hc * 2, 2, 2),       # 112x112 --> 56x56
            self.block(hc * 2, hc * 4, 2, 2),   # 56x56 --> 28x28
            self.block(hc * 4, hc * 8, 2, 2),   # 28x28 --> 14x14
            self.block(hc * 8, hc * 16, 2, 2),  # 14x14 --> 7x7
            self.block(hc * 16, hc * 32, 2, 2),  # 7x7 --> 3x3
            self.block(hc * 32, 1, 2, 2, final_layer=True),  # 3x3 --> 1x1
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=4,
        stride: int=2,
        final_layer: bool=False,
    ) -> nn.Sequential:
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.LeakyReLU(negative_slope=0.2) # ! ?Not sure
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Function for completing a forward pass of the discriminator. Given an image tensor it returns a 1-dimensional tensor representing real or fake.

        Args:
            image (torch.Tenor): Real of fake image.

        Returns:
            torch.Tensor: Probability that given image is real or fake. Shape: [batch_size, 1].
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

def test_discriminator_output_shape() -> None:
    """Function tests if the output shape matches the desired one."""
    disc = Discriminator300().to(c.DEVICE)
    height, width = c.RESIZE_TO_SHAPE
    image = torch.randn(c.BATCH_SIZE, c.IMAGE_CHANNELS, height, width).to(c.DEVICE)
    output = disc(image)
    assert output.shape == torch.Size([c.BATCH_SIZE, 1])


if __name__ == "__main__":
   test_discriminator_output_shape()