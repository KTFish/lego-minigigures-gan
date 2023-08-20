import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_channels: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = self.make_conv_block(in_channels, hidden_channels, 3, 1, 1)
        self.conv2 = self.make_conv_block(hidden_channels, hidden_channels, 3, 1, 1)
        self.conv3 = self.make_conv_block(hidden_channels, hidden_channels, 3, 1, 1)
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 32 * hidden_channels, out_channels)
        )

    def make_conv_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x


def test_cnn() -> None:
    model = CNN()
    X = torch.rand(32, 3, 32, 32)
    pred = model(X)

    assert pred.shape == torch.Size([32, 1])


if __name__ == "__main__":
    test_cnn()
