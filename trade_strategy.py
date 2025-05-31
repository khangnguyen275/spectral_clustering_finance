lookback_window = 60
lookforward_window = 3
w = 5
eligible_dates_txt_output = path + '/eligible_dates.txt'
eligible_dates = get_eligible_date_paths_from_file(eligible_dates_txt_output)

def execute_trading_strategy(win_threshold: float,
                             lookback_window = 60,
                             lookforward_window = 3,
                             w = 5):
  # record the total number of days
  num_dates = len(eligible_dates)
  # first trading day
  current_date = lookback_window
  # record daily_PnL
  daily_PnL = []

  # renew_portfolio criteria
  trading_period = 0
  trading_PnL = 0
  update_portfolio = True

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
    R_cov = clusterize('SPONGE', 'mar-pa', R_cov, market_cov)
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