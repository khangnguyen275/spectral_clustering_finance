import pandas as pd
import numpy as np

from utils.cluster import clusterize
from utils.returns import get_sliding_window_data

def identify_stocks(R_curr: pd.DataFrame):
    R_curr = R_curr.copy()

    # On each day, calculate the deviation from the mean of the cluster for each stock
    numeric_cols = R_curr.columns.difference(['ticker', 'cluster'])
    R_curr[numeric_cols] = (
        R_curr
        .groupby('cluster')[numeric_cols]
        .transform(lambda col: col - col.mean())
    )

    # Drop the non-numeric columns, then sum all deviations within the sliding window
    R_curr['deviation'] = R_curr.drop(columns=['ticker', 'cluster']).sum(axis=1)

    # Identify winners and losers based on the threshold value p
    threshold = 0

    # Start with zeros
    R_curr['trade'] = 0

    # Set +1 where deviation exceeds threshold signifying winners
    R_curr.loc[R_curr['deviation'] > threshold, 'trade'] = 1

    # Set -1 where deviation is below threshold signifying losers
    R_curr.loc[R_curr['deviation'] < threshold, 'trade'] = -1

    return R_curr

def assign_stock_weights(R_curr: pd.DataFrame):

  R_curr = R_curr.copy()

  # Calculate nK and mK for each cluster
  cluster_counts = R_curr.groupby('cluster')['trade'].value_counts().unstack(fill_value=0)

  # Calculate the notional values
  # If trade is +1, notional is 1 / number of +1 trades in the cluster
  # If trade is -1, notional is 1 / number of -1 trades in the cluster
  # If trade is 0, notional is 0
  R_curr['notional'] = 0.0  # Initialize the 'notional' column with zeros

  for cluster_id in cluster_counts.index:
      nK = cluster_counts.loc[cluster_id].get(1, 0)  # Get count of +1 trades, default to 0 if no +1
      mK = cluster_counts.loc[cluster_id].get(-1, 0) # Get count of -1 trades, default to 0 if no -1

      # Assign notional for +1 trades in the current cluster, handling division by zero
      if nK > 0:
          R_curr.loc[(R_curr['cluster'] == cluster_id) & (R_curr['trade'] == 1), 'notional'] = -1 / nK

      # Assign notional for -1 trades in the current cluster, handling division by zero
      if mK > 0:
          R_curr.loc[(R_curr['cluster'] == cluster_id) & (R_curr['trade'] == -1), 'notional'] = 1 / mK

  # Display the updated DataFrame
  return R_curr

def execute_trading_strategy(win_threshold: float,
                            lookback_window = 60,
                            lookforward_window = 3,
                            w = 5,
                            eligible_dates = None,
                            cl_med = 'SPONGE',
                            num_med = 'var'):
    # record the total number of days
    num_dates = len(eligible_dates)
    # first trading day
    current_date = lookback_window
    # record daily_PnL
    daily_PnL = []

# renew_portfolio criteria
    # trading_period = 0
    # trading_PnL = 0
    # update_portfolio = True

    while current_date + lookforward_window < 200:
        start_date = current_date - lookback_window
        # size of R_curr: #stocks x (1 ticker + 63 days)
        # size of market_curr: 63
        R_curr, market_curr = get_sliding_window_data(eligible_dates = eligible_dates,
                                                    lookback_window = lookback_window,
                                                    lookforward_window=lookforward_window,
                                                    start_date = start_date)
        # R_cov is the matrix containing the return in the 60 days lookback window
        R_cov = R_curr.iloc[:, : lookback_window + 1]
        market_cov = market_curr[: lookback_window]
        # clusterize the stocks
        R_cov = clusterize(cl_med, num_med, R_cov, market_cov)
        R_cov = assign_stock_weights(identify_stocks(R_cov))
        # calculate PnLs for the lookforward window
        bet_size = R_cov['notional'].to_numpy()
        lookback = -1 * lookforward_window
        future_return = R_curr.iloc[:, lookback:].to_numpy()
        PnLs = future_return.T @ bet_size
        Cumpnl = np.cumsum(PnLs)

        if np.max(Cumpnl) > win_threshold:
            index = np.argmax(Cumpnl > win_threshold)
            PnLs = PnLs[:index+1]

            daily_PnL += PnLs.tolist()
            current_date += index + 1
        else:
            daily_PnL += PnLs.tolist()
            current_date += lookforward_window
        print(f"Day {current_date+1}: PnL = {PnLs}")
    return daily_PnL