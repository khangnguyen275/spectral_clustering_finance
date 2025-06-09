import pandas as pd
import numpy as np

from utils.cluster import *
from utils.returns import *
from utils.helper import *

def identify_stocks(R_curr: pd.DataFrame, lookforward_window = 3, w = 5, threshold = 0):
    R_curr = R_curr.copy()

    # On each day, calculate the deviation from the mean of the cluster for each stock
    numeric_cols = R_curr.columns.difference(['ticker', 'cluster'])
    R_curr[numeric_cols] = (
        R_curr
        .groupby('cluster')[numeric_cols]
        .transform(lambda col: col - col.mean())
    )
    # Drop the non-numeric columns, then sum all deviations within the specified window
    # Drop the non-numeric columns, then sum all deviations within the specified window
    numeric_cols = R_curr.columns.difference(['ticker', 'cluster'])
    # Select the columns in the window: from -lookforward_window-w to -lookforward_window (exclusive)
    window_start = -lookforward_window - w
    window_end = -lookforward_window
    window_cols = numeric_cols[window_start:window_end]
    R_curr['deviation'] = R_curr[window_cols].sum(axis=1)
    # Drop the non-numeric columns, then sum all deviations within the sliding window
    #R_curr['deviation'] = R_curr.drop(columns=['ticker', 'cluster']).sum(axis=1)

    # Start with zeros
    R_curr['trade'] = 0

    # Set +1 where deviation exceeds threshold signifying winners
    R_curr.loc[R_curr['deviation'] > threshold, 'trade'] = 1

    # Set -1 where deviation is below threshold signifying losers
    R_curr.loc[R_curr['deviation'] < - threshold, 'trade'] = -1

    return R_curr


# def identify_stocks(R_curr: pd.DataFrame):
#     R_curr = R_curr.copy()

#     # On each day, calculate the deviation from the mean of the cluster for each stock
#     numeric_cols = R_curr.columns.difference(['ticker', 'cluster'])
#     R_curr[numeric_cols] = (
#         R_curr
#         .groupby('cluster')[numeric_cols]
#         .transform(lambda col: col - col.mean())
#     )

#     # Drop the non-numeric columns, then sum all deviations within the sliding window
#     R_curr['deviation'] = R_curr.drop(columns=['ticker', 'cluster']).sum(axis=1)

#     # Identify winners and losers based on the threshold value p
#     threshold = 0

#     # Start with zeros
#     R_curr['trade'] = 0

#     # Set +1 where deviation exceeds threshold signifying winners
#     R_curr.loc[R_curr['deviation'] > threshold, 'trade'] = 1

#     # Set -1 where deviation is below threshold signifying losers
#     R_curr.loc[R_curr['deviation'] < threshold, 'trade'] = -1

#     return R_curr

def discard_bottom_quarter_clusters(corr_matrix, cluster_vector):
    """
    Discards the bottom 10 clusters with the lowest correlation_object_function term values.
    Returns a mask (boolean array) indicating which stocks belong to the selected clusters.
    """
    unique_clusters = np.unique(cluster_vector)
    # Compute the term for each cluster
    cluster_terms = []
    for cluster_idx in unique_clusters:
        mask = (cluster_vector == cluster_idx)
        cluster_size = np.sum(mask)
        if cluster_size > 1:
            term = (np.sum(corr_matrix[mask][:, mask]) - cluster_size) / (cluster_size ** 2 - cluster_size)
        else:
            term = np.nan  # Avoid division by zero for singleton clusters
        cluster_terms.append((cluster_idx, term))
    # Sort clusters by term (ascending, lower is worse)
    sorted_clusters = sorted([ct for ct in cluster_terms if not np.isnan(ct[1])], key=lambda x: x[1])
    # Discard bottom 1/4 clusters
    n_discard = len(sorted_clusters) // 4
    selected_clusters = set([cl for cl, _ in sorted_clusters[n_discard:]])
    # Create mask for stocks in selected clusters
    selected_mask = np.array([cl in selected_clusters for cl in cluster_vector])
    return selected_mask, selected_clusters

def assign_stock_weights(R_curr: pd.DataFrame, weighting_scheme='uniform'):
    R_curr = R_curr.copy()
    R_curr['notional'] = 0.0  # Initialize the 'notional' column with zeros
    
    if weighting_scheme == 'uniform':
        # Calculate nK and mK for each cluster
        cluster_counts = R_curr.groupby('cluster')['trade'].value_counts().unstack(fill_value=0)

        # Calculate the notional values
        # If trade is +1, notional is 1 / number of +1 trades in the cluster
        # If trade is -1, notional is 1 / number of -1 trades in the cluster
        # If trade is 0, notional is 0
        for cluster_id in cluster_counts.index:
            nK = cluster_counts.loc[cluster_id].get(1, 0)  # Get count of +1 trades, default to 0 if no +1
            mK = cluster_counts.loc[cluster_id].get(-1, 0) # Get count of -1 trades, default to 0 if no -1

            # Assign notional for +1 trades in the current cluster, handling division by zero
            if nK > 0:
                R_curr.loc[(R_curr['cluster'] == cluster_id) & (R_curr['trade'] == 1), 'notional'] = -1 / nK

            # Assign notional for -1 trades in the current cluster, handling division by zero
            if mK > 0:
                R_curr.loc[(R_curr['cluster'] == cluster_id) & (R_curr['trade'] == -1), 'notional'] = 1 / mK

    elif weighting_scheme == 'linear':
        # Apply linear weighting within each cluster
        for cluster_id in R_curr['cluster'].unique():
            cluster_mask = R_curr['cluster'] == cluster_id

            # Winners in this cluster
            winners = R_curr[cluster_mask & (R_curr['trade'] == 1)]
            x_values = winners['deviation']
            sum_x = x_values.sum()
            if sum_x != 0:
                R_curr.loc[cluster_mask & (R_curr['trade'] == 1), 'notional'] = -x_values / sum_x
            else:
                R_curr.loc[cluster_mask & (R_curr['trade'] == 1), 'notional'] = 0

            # Losers in this cluster
            losers = R_curr[cluster_mask & (R_curr['trade'] == -1)]
            y_values = -losers['deviation']
            sum_y = y_values.sum()
            if sum_y != 0:
                R_curr.loc[cluster_mask & (R_curr['trade'] == -1), 'notional'] = y_values / sum_y
            else:
                R_curr.loc[cluster_mask & (R_curr['trade'] == -1), 'notional'] = 0
            R_curr.loc[cluster_mask & (R_curr['trade'] == 0), 'notional'] = 0
    
    elif weighting_scheme == 'exponential':
        # Apply exponential weighting within each cluster
        for cluster_id in R_curr['cluster'].unique():
            cluster_mask = R_curr['cluster'] == cluster_id

            # Winners in this cluster
            winners = R_curr[cluster_mask & (R_curr['trade'] == 1)]
            x_values = winners['deviation']
            exp_x = np.exp(x_values)
            sum_exp_x = exp_x.sum()
            if sum_exp_x != 0:
                R_curr.loc[cluster_mask & (R_curr['trade'] == 1), 'notional'] = -exp_x / sum_exp_x
            else:
                R_curr.loc[cluster_mask & (R_curr['trade'] == 1), 'notional'] = 0

            # Losers in this cluster
            losers = R_curr[cluster_mask & (R_curr['trade'] == -1)]
            y_values = -losers['deviation']
            exp_y = np.exp(y_values)
            sum_exp_y = exp_y.sum()
            if sum_exp_y != 0:
                R_curr.loc[cluster_mask & (R_curr['trade'] == -1), 'notional'] = exp_y / sum_exp_y
            else:
                R_curr.loc[cluster_mask & (R_curr['trade'] == -1), 'notional'] = 0

            R_curr.loc[cluster_mask & (R_curr['trade'] == 0), 'notional'] = 0
    
    elif weighting_scheme == 'threshold':
        pass

    return R_curr

def execute_trading_strategy(win_threshold: float,
                            lookback_window = 60,
                            lookforward_window = 3,
                            w = 5,
                            eligible_dates = None,
                            cl_med = 'SPONGE',
                            num_med = 'var',
                            winsorize_raw = False,
                            winsorize_res = False,
                            winsor_param = 0.05,
                            weighting_scheme = 'uniform',
                            cluster_selection = False,
                            num_dates = None):
    # record the total number of days
    if num_dates is None:
        num_dates = len(eligible_dates)
    
    # first trading day
    current_date = lookback_window
    # record daily_PnL
    daily_PnL = []
    curr_date_str = []
    success = 0
    num_period = 0

# renew_portfolio criteria
    # trading_period = 0
    # trading_PnL = 0
    # update_portfolio = True

    # while current_date + lookforward_window < num_dates:
    while current_date + lookforward_window < 200:
        num_period += 1
        start_date = current_date - lookback_window
        # size of R_curr: #stocks x (1 ticker + 63 days)
        # size of market_curr: 63
        R_curr, market_curr = get_sliding_window_data(eligible_dates = eligible_dates,
                                                    lookback_window = lookback_window,
                                                    lookforward_window=lookforward_window,
                                                    start_date = start_date)
        
        if winsorize_raw:
            R_curr_np = R_curr.select_dtypes(include='number').to_numpy()
            for j in range(R_curr_np.shape[1]):
                R_curr_np[:,j] = winsorize(R_curr_np[:,j], winsor_param, 1-winsor_param)
            R_curr.iloc[:, 1:] = R_curr_np
        
        # R_cov is the matrix containing the return in the 60 days lookback window
        R_cov = R_curr.iloc[:, : lookback_window + 1]
        market_cov = market_curr[: lookback_window]
        # clusterize the stocks
        R = R_cov.copy()
        market = market_cov.copy()
        residual_returns_matrix = get_market_residual_returns(R, market)
        residual_returns_matrix = residual_returns_matrix.astype(float).T
        corr = compute_correlation_matrix(residual_returns_matrix)
        
        R_cov = clusterize(cl_med, num_med, R_cov, market_cov, winsorize_res, winsor_param)
        
        if cluster_selection:
            selected_mask, selected_clusters = discard_bottom_quarter_clusters(corr, R_cov['cluster'].values)
            R_cov = R_cov[selected_mask].reset_index(drop=True)
            R_cov = assign_stock_weights(identify_stocks(R_cov))
            # calculate PnLs for the lookforward window
            bet_size = R_cov['notional'].to_numpy()
            lookback = -1 * lookforward_window
            future_return = R_curr.iloc[:, lookback:].to_numpy()
            future_return = future_return[selected_mask, :]
            
            
        else:
            R_cov = assign_stock_weights(identify_stocks(R_cov), weighting_scheme = weighting_scheme)
            # calculate PnLs for the lookforward window
            bet_size = R_cov['notional'].to_numpy()
            lookback = -1 * lookforward_window
            future_return = R_curr.iloc[:, lookback:].to_numpy()


        PnLs = future_return.T @ bet_size
        num_clusters = R_cov['cluster'].nunique()
        PnLs = PnLs / (2 * num_clusters)
        Cumpnl = np.cumsum(PnLs)
        

        if np.max(Cumpnl) > win_threshold:
            success += 1
            index = np.argmax(Cumpnl > win_threshold)
            PnLs = PnLs[:index+1]

            daily_PnL += PnLs.tolist()
            current_date += index + 1
            # record date stamps 
            for i in range(index+1):
                curr_date_str.append(R_curr.columns[-lookforward_window+i])
            
        else:
            daily_PnL += PnLs.tolist()
            current_date += lookforward_window
            # record date stamps
            for i in range(lookforward_window):
                curr_date_str.append(R_curr.columns[-lookforward_window+i])
        print(f"Day {current_date+1}: PnL = {PnLs}")
    date = pd.to_datetime(curr_date_str, format='%Y%m%d')

    return daily_PnL, date, float(success) / float(num_period)