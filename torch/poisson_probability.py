import torch

def poisson_probability(k, lam):
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    :return: Probability of k events occurring, rounded to 5 decimal places
    """
    lam_t = torch.tensor(float(lam))
    k_t = torch.tensor(float(k))

    dist = torch.distributions.Poisson(lam_t)

    # Compute log probability for numerical stability
    log_prob = dist.log_prob(k_t)
    probability = torch.exp(log_prob)

    return round(probability.item(), 5)
