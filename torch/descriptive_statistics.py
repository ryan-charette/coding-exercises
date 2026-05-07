import torch

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles, and interquartile range (IQR)
    """
    values = torch.as_tensor(data, dtype=torch.float32)

    mean = torch.mean(values)

    median = torch.quantile(values, 0.50)

    unique_values, counts = torch.unique(values, return_counts=True)
    max_count_index = torch.argmax(counts)
    mode = unique_values[max_count_index]

    variance = torch.var(values, correction=0)
    standard_deviation = torch.std(values, correction=0)

    percentile_25 = torch.quantile(values, 0.25)
    percentile_50 = torch.quantile(values, 0.50)
    percentile_75 = torch.quantile(values, 0.75)

    interquartile_range = percentile_75 - percentile_25

    statistics = {
        "mean": mean.item(),
        "median": median.item(),
        "mode": mode.item(),
        "variance": variance.item(),
        "standard_deviation": standard_deviation.item(),
        "25th_percentile": percentile_25.item(),
        "50th_percentile": percentile_50.item(),
        "75th_percentile": percentile_75.item(),
        "interquartile_range": interquartile_range.item(),
    }

    return statistics
