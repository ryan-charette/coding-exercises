import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles, and interquartile range
    """
    values = np.array(data)

    mean = np.mean(values)
    median = np.median(values)

    unique_values, counts = np.unique(values, return_counts=True)
    max_count_index = np.argmax(counts)
    mode = unique_values[max_count_index].item()

    variance = np.var(values)
    standard_deviation = np.std(values)

    percentile_25 = np.percentile(values, 25)
    percentile_50 = np.percentile(values, 50)
    percentile_75 = np.percentile(values, 75)

    interquartile_range = percentile_75 - percentile_25

    statistics = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "25th_percentile": percentile_25,
        "50th_percentile": percentile_50,
        "75th_percentile": percentile_75,
        "interquartile_range": interquartile_range,
    }

    return statistics
