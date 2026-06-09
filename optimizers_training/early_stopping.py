import torch
from typing import Tuple

def early_stopping(val_losses: torch.Tensor, patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine when to stop training early based on validation losses.
    
    Args:
        val_losses: A 1D tensor of validation losses for each epoch
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum decrease in loss to qualify as an improvement
    
    Returns:
        Tuple of (stop_epoch, best_epoch)
    """
    pass
