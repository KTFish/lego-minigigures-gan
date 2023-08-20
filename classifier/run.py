# These scripts are desgined to train a CNN Classifier (not a GAN) on the custom Lego Minifigures Dataset.
# It is a binary classifier recognizing between two classes: Lego and NOT Lego.

import torch
import config
import dataset
from train import train
from torch import nn
from model import CNN


def run():
    train_loader, test_loader = dataset.get_dataloaders(batch_size=config.BATCH_SIZE)

    model = CNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
    train(model, train_loader, test_loader, optimizer, criterion)


if __name__ == "__main__":
    run()
