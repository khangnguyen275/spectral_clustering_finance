import numpy as np
from signet.cluster import Cluster
import signet.block_models as bm
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import csc_matrix
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns

def compute_correlation_matrix(residual_returns_matrix, w = 5):
    """
    Computes the correlation matrix from a matrix of residual returns.

    Args:
        residual_returns_matrix (np.ndarray): A w x N matrix of residual returns,
            where w is the number of days and N is the number of stocks.

    Returns:
        np.ndarray: An N x N correlation matrix.
    """
    # Take the day that we're on, subtract by w days to get the start date,
    # and then take the period in the next w days
    residual_returns_matrix = residual_returns_matrix[-w : , :]
    # Center the residuals: subtract the mean of each column (stock)
    residuals_centered = residual_returns_matrix - np.mean(residual_returns_matrix, axis=0)

    # Compute the sample covariance matrix (N x N)
    covariance_matrix = residuals_centered.T @ residuals_centered / (residual_returns_matrix.shape[0] - 1)

    # Compute standard deviations
    std_devs = np.std(residual_returns_matrix, axis=0, ddof=1)

    # Avoid division by zero by setting zero stds to 1 temporarily (will fix values after)
    std_devs_safe = np.where(std_devs == 0, 1, std_devs)


    # Compute outer product of stds for normalization
    std_matrix = np.outer(std_devs_safe, std_devs_safe)

    # Compute correlation matrix
    correlation_matrix = covariance_matrix / std_matrix

    # Set any entries where std was zero to zero correlation
    zero_std_mask = (std_devs == 0)
    correlation_matrix[zero_std_mask, :] = 0
    correlation_matrix[:, zero_std_mask] = 0

    return correlation_matrix

def plot_correlation_matrix(correlation_matrix, title='Correlation Matrix Heatmap'):
    """
    Plots the correlation matrix using a heatmap.

    Args:
        correlation_matrix (np.ndarray): The correlation matrix to plot.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, square=True, fmt=".2f")
    plt.title(title)
    plt.xlabel('Stocks')
    plt.ylabel('Stocks')
    plt.show()

