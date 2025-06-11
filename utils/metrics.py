import numpy as np
import pandas as pd

def calculate_daily_PnL(R_curr: pd.DataFrame, window_width: int):
    """
    This function is deprecated. Calculates the daily Profit and Loss (PnL) based on the provided returns DataFrame.
    Args:
        R_curr (pd.DataFrame): DataFrame containing at least the columns for returns, 'notional', and 'cluster'.
        window_width (int): The index or column position to select the daily return from R_curr.
    Returns:
        float: The calculated daily PnL, normalized by twice the number of unique clusters.
    Notes:
        - Assumes that the column at position `window_width` in R_curr contains the daily returns.
        - The 'notional' column represents the bet size for each entry.
        - The 'cluster' column is used to determine the number of unique clusters for normalization.
    """
    
    daily_return = R_curr[window_width]
    daily_return = daily_return.to_numpy()[:, 1]
    bet_size = R_curr['notional'].to_numpy()
    num_clusters = R_curr['cluster'].nunique()
    PnL = np.dot(daily_return, bet_size)/(2 * num_clusters)
    return PnL

def calculate_Return_Rate(R_curr: pd.DataFrame, window_width: int):
    """
    Calculate the return rate (PnL) for a given DataFrame of returns, bet sizes, and clusters.
    Args:
        R_curr (pd.DataFrame): DataFrame containing at least the following columns:
            - an integer-indexed column for daily returns (accessed via window_width),
            - 'notional' for bet sizes,
            - 'cluster' for cluster assignments.
        window_width (int): The column index in R_curr to use for daily returns.
    Returns:
        float: The calculated profit and loss (PnL) normalized by twice the number of clusters.
    Notes:
        - Assumes that the daily return column is at position `window_width` in the DataFrame.
        - The function multiplies daily returns by bet sizes, sums them, and normalizes by 2 times the number of unique clusters.
    """
    
    daily_return = R_curr[window_width]
    daily_return = daily_return.to_numpy()[:, 1]
    bet_size = R_curr['notional'].to_numpy()
    num_clusters = R_curr['cluster'].nunique()
    PnL = np.dot(daily_return, bet_size)/(2 * num_clusters)
    return PnL

def calculate_Sharpe_Ratio(PnLs):
    """Calculate annualized Sharpe Ratio for a series of PnL values."""
    return (np.mean(PnLs) / np.std(PnLs)) * np.sqrt(252)