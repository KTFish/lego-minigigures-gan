
import torch
from torch import nn

# Data
RESIZE_TO_SHAPE = (28, 28)

# Hyper parameters
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
BATCH_SIZE = 128

Z_DIM = 10
HIDDEN_CHANNELS_GEN = 64
HIDDEN_CHANNELS_DISC = 16
IMAGE_CHANNELS = 3 # Number of channels in the image (here 3, because it is a RGB image).

# Training
NUM_EPOCHS = 100
DISPLAY_STEP = 50 # !
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CRITERION = nn.BCEWithLogitsLoss()

