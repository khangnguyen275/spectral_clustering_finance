import os
import pandas as pd
import numpy as np


def get_eligible_date_paths_from_file(file_path):
    """
    Reads a file containing file paths (one per line), strips whitespace from each line,
    and returns a list of cleaned file paths.

    Args:
        file_path (str): The path to the file containing the list of file paths.

    Returns:
        list: A list of strings, each representing a cleaned file path from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If any other error occurs during file reading.

    Side Effects:
        Prints the number of paths read from the file or error messages if exceptions occur.
    """
    eligible_dates_from_file = []
    # Open the file in read mode ('r')
    try:
        with open(file_path, 'r') as f:
            # Read each line from the file
            for line in f:
                # Remove any leading or trailing whitespace (including the
                # newline character)
                cleaned_line = line.strip()
                # Add the cleaned line (path) to the list
                eligible_dates_from_file.append(cleaned_line)

        print(
            f"Successfully read {
                len(eligible_dates_from_file)} paths from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return eligible_dates_from_file


def construct_short_price_matrix(eligible_dates,
                                lookback_window=60,
                                lookforward_window=3,
                                start_date=0):
    """
    Constructs a cleaned price matrix for a short window of dates from a list of eligible date file paths.
    This function reads compressed CSV files containing stock price data for a sequence of dates, extracts the 'close' prices for each ticker, and constructs a merged DataFrame where each column corresponds to a date and each row to a ticker. The function applies several filters:
    - Only keeps the 'ticker' and 'close' columns from each file.
    - Removes all rows below 'SPY' and above the first ticker starting with 'A' in each file.
    - Renames the 'close' column to the corresponding date (extracted from the file name).
    - Merges all dates' close prices into a single DataFrame, keeping only tickers present in all dates.
    - Moves the 'SPY' row to the top of the DataFrame.
    - Drops any rows with missing values.
    Args:
        eligible_dates (list of str): List of file paths to the compressed CSV files, each representing a date.
        lookback_window (int, optional): Number of days to look back from the start date. Defaults to 60.
        lookforward_window (int, optional): Number of days to look forward from the start date. Defaults to 3.
        start_date (int, optional): Index in eligible_dates to start the window. Defaults to 0.
    Returns:
        pandas.DataFrame: Cleaned DataFrame of close prices, indexed by ticker, with columns for each date in the window.
    """
    
    # get the paths to the dates in the lookback_window + lookforward_window +
    # 1 day buffer
    end_date = start_date + lookback_window + lookforward_window
    price_window_paths = eligible_dates[start_date: end_date + 1]
    close_price = pd.DataFrame()
    for path in price_window_paths:
        # read the coresponding data frame
        df = pd.read_csv(path, compression='gzip')
        # dropping all columns but `ticker` and `close`
        columns_to_drop = [
            col for col in df.columns if (
                col not in [
                    'close', 'ticker'])]
        df = df.drop(columns=columns_to_drop)
        # remove all rows below `SPY` and above the first ticker that starts
        # with `A`
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
            close_price = pd.merge(close_price, df, on='ticker', how='inner')

    # Separate the row where ticker is 'SPY'
    spy_row = close_price[close_price['ticker'] == 'SPY']

    # Separate the rest of the DataFrame (excluding the SPY row)
    other_rows = close_price[close_price['ticker'] != 'SPY']

    # Concatenate the SPY row with the rest of the DataFrame
    close_price = pd.concat([spy_row, other_rows])

    # Drop rows with any NaN values
    close_price_cleaned = close_price.dropna(axis=0)

    return close_price_cleaned


def construct_return_df(close_price, return_type='linear'):
    """
    Constructs a DataFrame of asset returns from a DataFrame of close prices.

    Parameters
    ----------
    close_price : pandas.DataFrame
        DataFrame containing asset close prices. Must include a 'ticker' column and one or more date columns.
    return_type : str, optional
        Type of return to compute: 'linear' for simple returns (default), or 'log' for logarithmic returns.

    Returns
    -------
    pandas.DataFrame
        DataFrame of returns with the same tickers and date columns (excluding the first date column, which will be NaN after differencing).
        The 'ticker' column is the first column, followed by return columns for each date (excluding the first date).

    Notes
    -----
    - The function drops the first date column from the returns, as it will contain NaN values due to the differencing operation.
    - The output DataFrame preserves the order of tickers and dates as in the input.
    - Requires NumPy and pandas.
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
    return_df = return_df.iloc[:, 1:]  # Drop the first column of returns

    # Add the 'ticker' column back to the log returns DataFrame
    return_df['ticker'] = close_price['ticker']

    # Reorder the columns to have 'ticker' as the first column
    # Get the original column order (excluding the first date which will have
    # NaNs)
    original_cols = close_price.columns.tolist()
    # Find the index of the first date column
    first_date_col_index = close_price.columns.get_loc(price_data.columns[0])
    # Keep columns from the second date column onwards
    date_cols_for_returns = original_cols[first_date_col_index + 1:]

    return_df = return_df[['ticker'] + date_cols_for_returns]
    return return_df


def clean_return_df(return_df,
                    drop_0_threshold=0.5,
                    drop_large_threshold=0.1,
                    large_return_threshold=1.0):
    """
    Cleans a DataFrame of ETF returns by removing rows (ETFs) that have too many zero or abnormally large returns.

    Args:
        return_df (pd.DataFrame): DataFrame containing ETF returns with a 'ticker' column and date columns as returns.
        drop_0_threshold (float, optional): Maximum allowed proportion of zero returns per ETF. Rows with a higher proportion are dropped. Default is 0.5.
        drop_large_threshold (float, optional): Maximum allowed proportion of large returns (greater than `large_return_threshold`) per ETF. Rows with a higher proportion are dropped. Default is 0.1.
        large_return_threshold (float, optional): Threshold above which a return is considered "large". Default is 1.0.

    Returns:
        pd.DataFrame: A cleaned DataFrame with ETFs (rows) that meet the specified criteria.
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
        total_days = returns.count()  # .count() excludes NaN values

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
    print("Original DataFrame shape:", return_df.shape)
    print("Cleaned DataFrame shape:", cleaned_return_df.shape)

    # print("\nCleaned Return Matrix (first 5 rows):")
    # print(cleaned_return_df.head())
    return cleaned_return_df


def get_sliding_window_data(eligible_dates,
                            lookback_window=60,
                            lookforward_window=3,
                            start_date=0):
    """
    Generates sliding window data for time series analysis.

    This function constructs a matrix of close prices over a specified lookback and lookforward window,
    computes returns, cleans the resulting DataFrame, and extracts the current market data and returns.

    Args:
        eligible_dates (list or array-like): List of dates eligible for constructing the sliding window.
        lookback_window (int, optional): Number of periods to look back for each window. Defaults to 60.
        lookforward_window (int, optional): Number of periods to look forward for each window. Defaults to 3.
        start_date (int, optional): Index or position to start the sliding window. Defaults to 0.

    Returns:
        tuple:
            - R_curr (pd.DataFrame): DataFrame containing the cleaned returns for all rows except the first.
            - market_curr (np.ndarray): Array containing the market data from the first row, excluding the 'ticker' column.
    """
    close_price = construct_short_price_matrix(
        eligible_dates,
        lookback_window=lookback_window,
        lookforward_window=lookforward_window,
        start_date=start_date)
    return_df = construct_return_df(close_price)
    cleaned_return_df = clean_return_df(return_df)
    market_curr = cleaned_return_df.iloc[0].drop(columns=['ticker']).values[1:]
    # Extract all rows except the first as a new DataFrame
    # .iloc[1:] selects all rows starting from the second row (index 1)
    # .copy() is used to avoid SettingWithCopyWarning
    R_curr = cleaned_return_df.iloc[1:].copy()
    return R_curr, market_curr

def get_market_residual_returns(R_curr, market_curr):
    """
    Computes the residual returns of assets after removing the linear effect of the market returns.

    Given a DataFrame of asset returns and a corresponding array of market returns, this function
    calculates the residual (idiosyncratic) returns for each asset by fitting a linear regression
    (using least squares) of each asset's returns against the market returns and subtracting the
    market component.

    Parameters
    ----------
    R_curr : pandas.DataFrame
        DataFrame containing asset returns. Must include a 'ticker' column, which will be dropped
        before computation. The remaining columns should represent return values over a sliding window.
    market_curr : numpy.ndarray or pandas.Series
        Array or Series of market returns corresponding to the same time window as the asset returns.

    Returns
    -------
    numpy.ndarray
        Array of residual returns for each asset, with the same shape as the input returns (excluding
        the 'ticker' column). Each row corresponds to an asset, and each column to a time point.

    Notes
    -----
    The function performs a vectorized calculation of the regression slope (beta) and intercept (alpha)
    for each asset, and returns the residuals (actual returns minus the market component).
    """
    R_curr_temp = R_curr.drop(columns=['ticker'])  # .iloc[:, :-3]

    # extract raw arrays
    R = R_curr_temp.values              # shape (df.shape[0], sliding_window)
    m = market_curr  # [:-3]                # shape (sliding_window,)

    # precompute market stats
    m_mean = m.mean()
    m_dev = m - m_mean
    den = np.dot(m_dev, m_dev)  # = Σ (m_j - m_mean)^2

    # row means of R
    R_mean = R.mean(axis=1)       # shape (df.shape[0],)

    # row-by-row covariance with market: Σ_i (R[i,j] - R_mean[i]) * (m[j] -
    # m_mean)
    num = (R - R_mean[:, None]) @ m_dev   # shape (df.shape[0],)

    # slopes β and intercepts α
    beta = num / den                       # shape (df.shape[0],)

    R_res = R - beta[:, None] * m[:, None].T

    return R_res
