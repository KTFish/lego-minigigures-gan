import torch

RESIZE_TO_SHAPE = (28, 28)

CHANNELS_IMG = 3

# Training
EPOCHS = 5
BATCH_SIZE = 128  # In paper a batch of 128 was used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
LEAKY_RELU_SLOPE = 0.2
HIDDEN_UNITS = 64
Z_DIM = 100


# Optimizer (for details see papers section IV "DETAILS OF ADVERSARIAL TRAINING")
# Oprimizer = ADAM
BETAS = (0.5, 0.999)
LEARNING_RATE = 0.0002
