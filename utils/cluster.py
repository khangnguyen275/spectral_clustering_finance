import numpy as np
from signet.cluster import Cluster
import signet.block_models as bm
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import csc_matrix
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .returns import get_market_residual_returns
from utils.helper import winsorize

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

def get_num_of_clusters(corr, thr):
    """
    Determines the minimum number of clusters (principal components) required to explain a given threshold of the total variance in a correlation matrix.

    Parameters:
        corr (np.ndarray): The correlation matrix (square, symmetric).
        thr (float): The threshold (between 0 and 1) representing the fraction of total variance to be explained.

    Returns:
        int: The minimum number of clusters (eigenvalues) needed to reach or exceed the specified threshold of explained variance.

    Notes:
        - Uses eigenvalue decomposition to compute explained variance.
        - Assumes that the input matrix is symmetric and positive semi-definite.
    """
    eigs = eigh(corr, eigvals_only=True)
    eigs = np.flip(np.sort(eigs))
    sum_of_eigs = np.sum(eigs)
    running = 0
    for i in range(len(eigs)):
        running += eigs[i]
        if running / sum_of_eigs >= thr:
            return i+1

def clusterize(cl_med: str, num_med: str, R_cov: pd.DataFrame, market_cov, winsorize_res, winsor_param, clustering_window=20, num_clusters=40):
    """
    Clusterizes assets based on their residual returns correlation structure using different clustering methods.
    Parameters
    ----------
    cl_med : str
        The clustering method to use. Supported values:
        - 'SPONGE': Use the SPONGE algorithm for clustering.
        - 'industry': (Commented out) Cluster based on industry data.
    num_med : str
        Method to determine the number of clusters. Supported values:
        - 'var': Use explained variance to determine the number of clusters.
        - 'mar-pa': Use the Marchenko-Pastur distribution to determine the number of clusters.
        - 'self': Use the user-specified number of clusters.
    R_cov : pd.DataFrame
        DataFrame containing asset returns (or residual returns) to be clustered.
    market_cov : pd.DataFrame or np.ndarray
        Data representing the market returns or market covariance, used to compute residual returns.
    winsorize_res : bool
        Whether to winsorize the residual returns before clustering.
    winsor_param : float
        The winsorization parameter (e.g., 0.01 for 1% winsorization).
    clustering_window : int, optional
        The window size (number of periods) to use for clustering, by default 20.
    num_clusters : int, optional
        The number of clusters to use if `num_med` is 'self', by default 40.
    Returns
    -------
    pd.DataFrame
        The input DataFrame `R_cov` with an added 'cluster' column indicating the cluster assignment for each asset.
    Notes
    -----
    - This is the main clustering function that can be used to cluster assets based on their residual returns.
    - If the clustering method is 'SPONGE', the function computes the correlation matrix of residual returns and applies the SPONGE algorithm.
    - The number of clusters can be determined by explained variance, the Marchenko-Pastur distribution, or set manually.
    - If winsorization is enabled, each asset's residual returns are winsorized before clustering.
    - If clustering cannot be performed (e.g., invalid number of clusters), all assets are assigned to a default cluster (-1).
    - The 'industry' clustering method is present in comments and not currently active.
    """
    
    R = R_cov.copy()
    market = market_cov.copy()

    # compute the correlation matrix used for clusterization
    residual_returns_matrix = get_market_residual_returns(R, market)
    residual_returns_matrix = residual_returns_matrix.astype(float).T
    
    if winsorize_res:
        for i in range(residual_returns_matrix.shape[0]):
            residual_returns_matrix[i, :] = winsorize(residual_returns_matrix[i, :], winsor_param, 1-winsor_param)
    
    corr = compute_correlation_matrix(residual_returns_matrix)

    # choose which clusterization method to use
    if cl_med == 'SPONGE':
        # determine the number of clusters for the SPONGE algorithm
        # 'var' means we use percent of explained variance
        # 'mar-pa' means we use the marchenko-pastur distribution
        if num_med == 'var' or num_med == 'mar-pa':
            RRT_num_clusters = residual_returns_matrix[-clustering_window :, :]
            cov = 1/(clustering_window) * (RRT_num_clusters.T @ RRT_num_clusters)
        if num_med == 'var':
            k = get_num_of_clusters(cov, 0.9)
        if num_med == 'mar-pa':
            num_of_stocks = RRT_num_clusters.shape[1]
            rho = num_of_stocks / clustering_window
            lambda_plus = (1 + np.sqrt(rho)) ** 2
            print(lambda_plus)
            eigs = eigh(cov, eigvals_only=True)
            print(eigs)
            plt.figure(figsize=(6, 4))
            plt.hist(eigs, bins=20, edgecolor='black', alpha=0.7)
            plt.title("Distribution of Eigenvalues")
            plt.xlabel("Eigenvalue")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            k = np.sum(eigs > lambda_plus)
            print(k)
        # 'self' means we pass in # of clusters ourselves
        if num_med == 'self':
            k = num_clusters

        # Defensive check: skip clustering if matrix is empty or k is invalid
        if residual_returns_matrix.shape[1] == 0 or k is None or k < 1 or k > residual_returns_matrix.shape[1]:
            print(f"Skipping clustering: shape={residual_returns_matrix.shape}, k={k}")
            R['cluster'] = -1  # or some default/fallback
            return R

        # split the correlation matrix into positive and negative parts
        G_plus = np.maximum(corr, 0)
        G_minus = np.maximum(-corr, 0)
        # call the SPONGE algorithm
        c = Cluster((csc_matrix(G_plus), csc_matrix(G_minus)))
        predictions = c.SPONGE(k=k, tau_p=1, tau_n=1, eigens=None, mi=None)
        # append predicted cluster assignments
        R['cluster'] = predictions

    # cluster stocks based on given industry data
    # any stock that does not belong to the given list of stock-industry pairs
    # is clustered into one single, separate cluster
    # if cl_med == 'industry':
    #   ticker_to_cluster = dict(zip(sector['SPY.1'], sector['0']))
    #   R['cluster'] = R['ticker'].map(ticker_to_cluster)
    #   R['cluster'] = pd.to_numeric(R['cluster'], errors='coerce').astype('Int64')
    #   max_cluster = R['cluster'].max()
    #   R['cluster'] = R['cluster'].fillna(max_cluster + 1).astype(int)
    return R