import torch

def dice_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Dice Score (SÃ¸rensen-Dice coefficient) for binary classification.

    Args:
        y_true: Binary tensor of true labels.
        y_pred: Binary tensor of predicted labels.

    Returns:
        Dice Score as a float rounded to 3 decimal places.
    """
    return round(res, 3)
