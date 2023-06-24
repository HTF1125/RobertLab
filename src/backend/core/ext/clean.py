import numpy as np
import pandas as pd


def clean_weights(
    weights: pd.Series, num_decimal: int = 4, num_iterations: int = 100
) -> pd.Series:
    """Clean weights based on the number decimals and maintain the total of weights.

    Args:
        weights (pd.Series): asset weights.
        decimals (int, optional): number of round decimals. Defaults to 4.

    Returns:
        pd.Series: cleaned asset weights.
    """
    # clip weight values by minimum and maximum.
    tot_weight = weights.sum().round(num_decimal)
    weights = weights.round(decimals=num_decimal)
    # repeat round and weight calculation.
    for _ in range(num_iterations):
        weights = weights / weights.sum() * tot_weight
        weights = weights.round(decimals=num_decimal)
        if weights.sum() == tot_weight:
            return weights
    # if residual remains after repeated rounding.
    # allocate the the residual weight on the max weight.
    residual = tot_weight - weights.sum()
    # !!! Error may occur when there are two max weights???
    weights.iloc[np.argmax(weights)] += np.round(residual, decimals=num_decimal)
    return weights
