import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple


def train_simple_cnn_with_backprop(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    learning_rate: float,
    kernel_size: int = 3,
    num_filters: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening,
    and a dense layer with softmax output using backpropagation via PyTorch autograd.

    Assumes X has shape (n_samples, height, width) for grayscale images and y is
    one-hot encoded with shape (n_samples, num_classes).

    Parameters:
    X            : torch.Tensor, input data of shape (n_samples, height, width)
    y            : torch.Tensor, one-hot encoded labels of shape (n_samples, num_classes)
    epochs       : int, number of training epochs
    learning_rate: float, learning rate for SGD weight updates
    kernel_size  : int, size of the square convolutional kernel
    num_filters  : int, number of filters in the convolutional layer

    Returns:
    W_conv   : torch.Tensor, trained conv kernel weights (num_filters, 1, kernel_size, kernel_size)
    b_conv   : torch.Tensor, trained conv bias         (num_filters,)
    W_dense  : torch.Tensor, trained dense weights     (flattened_size, num_classes)
    b_dense  : torch.Tensor, trained dense bias        (num_classes,)
    '''
    pass
