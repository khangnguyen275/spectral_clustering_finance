import pandas as pd
import numpy as np

def construct_short_price_matrix(eligible_dates,
                               lookback_window=60,
                               lookforward_window=3,
                               start_date=0):
  """
  Constructs a cleaned price matrix for a short window of dates, focusing on tickers between 'SPY' and the first ticker starting with 'A'.
  This function reads compressed CSV files containing price data for a sequence of dates, extracts the 'close' prices for each ticker, 
  and merges them into a single DataFrame. It specifically retains only the rows for 'SPY' and all tickers that appear after 'SPY' 
  but before the first ticker starting with 'A' (exclusive). The resulting DataFrame is cleaned by removing any rows with missing values.
  Args:
    eligible_dates (list of str): List of file paths to the daily price data CSV files, ordered by date.
    lookback_window (int, optional): Number of days to look back from the start_date. Defaults to 60.
    lookforward_window (int, optional): Number of days to look forward from the start_date. Defaults to 3.
    start_date (int, optional): Index in eligible_dates to start the window. Defaults to 0.
  Returns:
    pandas.DataFrame: A cleaned DataFrame where each row is a ticker (with 'SPY' as the first row), 
              columns are dates (as extracted from file names), and values are close prices. 
              All rows with any missing values are dropped.
  """
  
  # get the paths to the dates in the lookback_window + lookforward_window + 1 day buffer
  end_date = start_date + lookback_window + lookforward_window
  price_window_paths = eligible_dates[start_date : end_date + 1]
  close_price = pd.DataFrame()
  for path in price_window_paths:
    # read the coresponding data frame
    df = pd.read_csv(path, compression='gzip')
    # dropping all columns but `ticker` and `close`
    columns_to_drop = [col for col in df.columns if (col not in ['close', 'ticker'])]
    df = df.drop(columns=columns_to_drop)
    # remove all rows below `SPY` and above the first ticker that starts with `A`
    for ticker in df['ticker']:
      if ticker == 'SPY':
        continue
      elif ticker.startswith('A'):
        break
      else:
        index_to_drop = df[df['ticker'] == ticker].index
        df = df.drop(index=index_to_drop)

    # rename the column `close` to the date
    parts = path.split('/')
    file_name = parts[-1]
    file_name = file_name.replace('.csv.gz', '')
    df = df.rename(columns={'close': file_name})
    df.set_index('ticker')

    # merge close price on current date into the close_price data frame
    if close_price.empty:
      close_price = df
    else:
      close_price = pd.merge(close_price, df, on='ticker', how = 'inner')

  # Separate the row where ticker is 'SPY'
  spy_row = close_price[close_price['ticker'] == 'SPY']

  # Separate the rest of the DataFrame (excluding the SPY row)
  other_rows = close_price[close_price['ticker'] != 'SPY']

  # Concatenate the SPY row with the rest of the DataFrame
  close_price = pd.concat([spy_row, other_rows])

  # Drop rows with any NaN values
  close_price_cleaned = close_price.dropna(axis=0)

  return close_price_cleaned

def construct_return_df(close_price, return_type = 'linear'):
  def construct_return_df(close_price, return_type='linear'):
    """
    Constructs a DataFrame of asset returns from a DataFrame of close prices.
    Parameters
    ----------
    close_price : pandas.DataFrame
      DataFrame containing asset close prices. Must include a 'ticker' column and one or more date columns.
    return_type : str, optional
      Type of return to compute: 
      - 'linear' for simple percentage returns (default)
      - 'log' for logarithmic returns
    Returns
    -------
    pandas.DataFrame
      DataFrame of returns with 'ticker' as the first column, followed by return columns for each date (excluding the first date, which will be dropped due to NaN values from differencing).
    Notes
    -----
    - The first date column is dropped from the returns DataFrame since returns cannot be computed for the initial date.
    - The function preserves the order of tickers and dates as in the input DataFrame.
    - Requires `numpy` as `np` if using log returns.
    """
  
  # Select only the date columns (exclude 'ticker')
  price_data = close_price.drop(columns=['ticker'])

  if return_type == 'linear':
    # linear return:
    return_df = price_data.pct_change(axis=1)
  elif return_type == 'log':
    # log return:
    return_df = np.log(price_data)
    return_df = return_df.diff(axis=1)

  # The first column after 'ticker' will have NaNs due to the diff operation.
  return_df = return_df.iloc[:, 1:] # Drop the first column of returns

  # Add the 'ticker' column back to the log returns DataFrame
  return_df['ticker'] = close_price['ticker']

  # Reorder the columns to have 'ticker' as the first column
  # Get the original column order (excluding the first date which will have NaNs)
  original_cols = close_price.columns.tolist()
  # Find the index of the first date column
  first_date_col_index = close_price.columns.get_loc(price_data.columns[0])
  # Keep columns from the second date column onwards
  date_cols_for_returns = original_cols[first_date_col_index + 1:]

  return_df = return_df[['ticker'] + date_cols_for_returns]
  return return_df

def clean_return_df(return_df,
                    drop_0_threshold = 0.5,
                    drop_large_threshold = 0.1,
                    large_return_threshold = 1.0):
  """
  Cleans a DataFrame of ETF returns by removing rows (ETFs) that have an excessive proportion of zero or abnormally large returns.
  Parameters
  ----------
  return_df : pd.DataFrame
    DataFrame contasining ETF returns. Must include a 'ticker' column and columns for each date with return values.
  drop_0_threshold : float, optional (default=0.5)
    The maximum allowed proportion of zero returns for an ETF. Rows with a higher proportion are removed.
  drop_large_threshold : float, optional (default=0.1)
    The maximum allowed proportion of large returns (greater than `large_return_threshold`) for an ETF. Rows with a higher proportion are removed.
  large_return_threshold : float, optional (default=1.0)
    The threshold above which a return is considered "large".
  Returns
  -------
  cleaned_return_df : pd.DataFrame
    A DataFrame with rows (ETFs) removed if they exceed the specified thresholds for zero or large returns.
  Notes
  -----
  - NaN values are ignored in the calculation of proportions.
  - The 'ticker' column is preserved in the output DataFrame.
  """
  # Get the date columns (excluding 'ticker')
  date_cols = return_df.columns.drop('ticker')

  # Create a boolean mask for rows to keep (initialized to True for all rows)
  rows_to_keep_mask = pd.Series(True, index=return_df.index)

  # Iterate through each row (ETF) in the DataFrame
  for index, row in return_df.iterrows():
      # Get the return values for the current ETF (excluding 'ticker')
      returns = row[date_cols]

      # Calculate the total number of days (non-NaN returns) for this ETF
      total_days = returns.count() # .count() excludes NaN values

      # Count the number of days with return equal to 0
      zero_return_count = (returns == 0).sum()

      # Count the number of days with return greater than 1
      large_return_count = (returns > large_return_threshold).sum()

      # Calculate the percentages
      zero_return_percentage = zero_return_count / total_days
      large_return_percentage = large_return_count / total_days

      # Check the conditions for removal
      if zero_return_percentage > drop_0_threshold or large_return_percentage > drop_large_threshold:
          rows_to_keep_mask[index] = False

  # Filter the DataFrame based on the mask
  cleaned_return_df = return_df[rows_to_keep_mask].copy()
  # print("Original DataFrame shape:", return_df.shape)
  # print("Cleaned DataFrame shape:", cleaned_return_df.shape)

  # print("\nCleaned Return Matrix (first 5 rows):")
  # print(cleaned_return_df.head())
  return cleaned_return_df

def get_sliding_window_data(eligible_dates,
                            lookback_window = 60,
                            lookforward_window = 3,
                            start_date = 0):
    """
    Generates sliding window return data and market returns for a given set of eligible dates.

    This function constructs a matrix of close prices using a sliding window approach,
    calculates returns, cleans the resulting DataFrame, and extracts both the market
    returns and the returns for individual assets.

    Args:
            eligible_dates (list or array-like): List of dates eligible for constructing the sliding window.
            lookback_window (int, optional): Number of periods to look back for the sliding window. Default is 60.
            lookforward_window (int, optional): Number of periods to look forward for the sliding window. Default is 3.
            start_date (int or str, optional): The starting date or index for the sliding window. Default is 0.

    Returns:
            tuple:
                    - R_curr (pd.DataFrame): DataFrame containing cleaned returns for all assets except the market.
                    - market_curr (np.ndarray): Array of market returns extracted from the first row of the cleaned DataFrame.

    Raises:
            ValueError: If input data is invalid or insufficient for the specified window sizes.

    Note:
            This function depends on the helper functions:
                    - construct_short_price_matrix
                    - construct_return_df
                    - clean_return_df
            Ensure these are defined and imported in the module.
    """
    close_price = construct_short_price_matrix(
        eligible_dates,
        lookback_window=lookback_window,
        lookforward_window=lookforward_window,
        start_date=start_date
    )
    return_df = construct_return_df(close_price)
    cleaned_return_df = clean_return_df(return_df)
    market_curr = cleaned_return_df.iloc[0].drop(labels=['ticker']).values
    # Extract all rows except the first as a new DataFrame
    R_curr = cleaned_return_df.iloc[1:].copy()
    return R_curr, market_curr

def get_market_residual_returns(R_curr, market_curr):
  """
  Calculates the market residual returns for a given set of stock returns
  and corresponding market returns.

  This function implements a simple linear regression model for each stock
  against the market to determine the portion of a stock's return that is
  not explained by the market movement (the residual).

  Args:
    R_curr (pd.DataFrame): A DataFrame containing stock returns for a
                           specific time window. Expected to have a 'ticker'
                           column and subsequent columns representing daily
                           returns for each stock.
    market_curr (np.ndarray): A NumPy array containing the market returns
                             (e.g., SPY returns) for the same time window.
                             Expected shape is (sliding_window,).

  Returns:
    np.ndarray: A NumPy array of the market residual returns.
                The shape is (number_of_stocks, sliding_window).
                Each row represents a stock, and each column represents a day
                within the sliding window.
  """
  R_curr_temp = R_curr.drop(columns=['ticker'])#.iloc[:, :-3]

  # extract raw arrays
  R = R_curr_temp.values              # shape (df.shape[0], sliding_window)
  m = market_curr#[:-3]                # shape (sliding_window,)

  # precompute market stats
  m_mean = m.mean()
  m_dev  = m - m_mean
  den    = np.dot(m_dev, m_dev)  # = Σ (m_j - m_mean)^2

  # row means of R
  R_mean = R.mean(axis=1)       # shape (df.shape[0],)

  # row-by-row covariance with market: Σ_i (R[i,j] - R_mean[i]) * (m[j] - m_mean)
  num    = (R - R_mean[:, None]) @ m_dev   # shape (df.shape[0],)

  # slopes β and intercepts α
  beta   = num / den                       # shape (df.shape[0],)

  R_res = R - beta[:, None] * m[:, None].T

  return R_res