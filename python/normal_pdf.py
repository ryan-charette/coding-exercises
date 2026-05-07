import math

def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    var = std_dev ** 2

	  exponent = -((x - mean) ** 2) / (2 * var)
    denominator = math.sqrt(2 * math.pi * var)

    pdf_val = math.exp(exponent) / denominator

    return round(pdf_val, 5)
