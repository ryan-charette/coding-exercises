import torch

def normal_pdf(x: torch.Tensor, mean: torch.Tensor, std_dev: torch.Tensor) -> float:
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated (torch.Tensor scalar).
    :param mean: The mean (mu) of the distribution (torch.Tensor scalar).
    :param std_dev: The standard deviation (sigma) of the distribution (torch.Tensor scalar).
    :return: The PDF value rounded to 5 decimal places.
    """
    distribution = torch.distributions.Normal(mean, std_dev)

    log_probability = distribution.log_prob(x)
    probability = torch.exp(log_probability)

    return round(probability.item(), 5)
