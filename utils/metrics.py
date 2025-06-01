import numpy as np
import pandas as pd

def calculate_daily_PnL(R_curr: pd.DataFrame, window_width: int):
    daily_return = R_curr[window_width]
    daily_return = daily_return.to_numpy()[:, 1]
    bet_size = R_curr['notional'].to_numpy()
    num_clusters = R_curr['cluster'].nunique()
    PnL = np.dot(daily_return, bet_size)/(2 * num_clusters)
    return PnL

def calculate_Return_Rate(R_curr: pd.DataFrame, window_width: int):
    daily_return = R_curr[window_width]
    daily_return = daily_return.to_numpy()[:, 1]
    bet_size = R_curr['notional'].to_numpy()
    num_clusters = R_curr['cluster'].nunique()
    PnL = np.dot(daily_return, bet_size)/(2 * num_clusters)
    return PnL