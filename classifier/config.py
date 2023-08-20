import torch

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
