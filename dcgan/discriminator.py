import torch
from torch import nn
import config as c

class Discriminator(nn.Module):
    def __init__(self, im_chan: int=c.IMAGE_CHANNELS, hidden_dim: int=c.HIDDEN_CHANNELS_DISC):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.block(im_chan, hidden_dim),
            self.block(hidden_dim, hidden_dim * 2),
            self.block(hidden_dim * 2, 1, final_layer=True),
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
                nn.Conv2d(in_channels, out_channels, kernel_size, stride)
                nn.LeakyReLU(negative_slope=0.2) # ! ?Not sure
            )

    def forward(self, image: torch.Tenor) -> torch.Tensor:
        """Function for completing a forward pass of the discriminator. Given an image tensor it returns a 1-dimensional tensor representing real or fake.

        Args:
            image (torch.Tenor): Real of fake image.

        Returns:
            torch.Tensor: Probability that given image is real or fake. Shape: [batch_size, 1].
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)