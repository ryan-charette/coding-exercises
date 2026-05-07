import torch
from torch.distributions import Binomial

def binomial_probability(n: int, k: int, p: float) -> torch.Tensor:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.

    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial

    Returns:
        Probability of k successes as a torch.Tensor scalar
    """
    total_count = torch.tensor(float(n))
    success_count = torch.tensor(float(k))
    probability_success = torch.tensor(float(p))

    distribution = Binomial(total_count=total_count, probs=probability_success)

    log_probability = distribution.log_prob(success_count)
    probability = torch.exp(log_probability)

    return probability
