import torch

# Configuration parameters
SOURCE_LENGTH = 262
TARGET_LENGTH = 189

def prepare_device():
    """
    Determines whether to use CUDA (GPU) or CPU based on availability and returns the appropriate device.

    Returns:
    torch.device: The device (CUDA or CPU) to be used for model operations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")