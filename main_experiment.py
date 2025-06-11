import sys
import os
import argparse
from utils.metrics import *
import csv
np.set_printoptions(precision=8, suppress=False)
possible_paths = [
        '/Users/khang/Desktop/math285j_project/data/drive-download-20250531T145738Z-1-001/CRSP Data Set',
        '/Users/lunjizhu/Desktop/MATH 285J Project Workspace/spectral_clustering_finance/data',
        'F:/spectral_clustering_finance/data/drive-download-20250531T145738Z-1-001/CRSP Data Set',
        '/Users/yifangu/Desktop/MATH 285J/285J Project/spectral_clustering_finance/data/CRSP Data Set'
]

path = None
for p in possible_paths:
    if os.path.isdir(p):
        path = p
        print(f"Using data path: {path}")
        break

if path is None:
    print("Error: No valid data path found.")
    sys.exit(1)
    
from utils.returns import *
from utils.trader import execute_trading_strategy
from utils.helper import *

parser = argparse.ArgumentParser(description="Spectral Clustering Finance Experiment")
# general parameters
parser.add_argument('--num_dates', type=int, default= -1, help='Number of eligible dates to use (default: -1, which means all dates)')
parser.add_argument('--num_med', type=str, choices=['self', 'var'], default='self', help="Number of medoids to use: 'self' or 'var' (default: 'self')")
parser.add_argument('--num_clusters', type=int, default=40, help='Number of clusters to use for clustering (default: 40)')
parser.add_argument('--win_threshold', type=float, default=0.001, help='Win threshold (non-negative float, default: 0.001)')
# cluster selection parameters
parser.add_argument('--cluster_selection', action='store_true', help='Enable cluster selection (default: False)')
parser.add_argument('--num_trading_clusters', type=int, default=40, help='Number of trading clusters to use (default: 40)')
# weighting parameters
parser.add_argument('--weight_type', choices=['uniform', 'linear', 'exponential'], default='uniform', help='Type of weighting to use (default: uniform)')
# winsorization parameters
parser.add_argument('--winsorize_raw', action='store_true', help='Enable windsorization for the raw returns (default: False)')
parser.add_argument('--winsorize_res', action='store_true', help='Enable windsorization for the res returns (default: False)')
parser.add_argument('--winsor_param', type=float, default=0.05, help='Winsorization parameter (default: 0.05)')

args = parser.parse_args()

num_dates = args.num_dates
num_med = args.num_med
num_clusters = args.num_clusters
win_threshold = args.win_threshold
cluster_selection = args.cluster_selection
num_trading_clusters = args.num_trading_clusters
weight_type = args.weight_type
winsorize_raw = args.winsorize_raw
winsorize_res = args.winsorize_res
winsor_param = args.winsor_param

if num_dates == -1:
    num_dates = None  # Use all dates if -1 is specified
num_med = args.num_med
print(f"cluster_selection: {cluster_selection}, weight_type: {weight_type}, windsorize_raw: {winsorize_raw}, windsorize_res: {winsorize_res}, winsor_param: {winsor_param}, num_dates: {num_dates}, num_med: {num_med}")

eligible_dates_txt_output = path + '/eligible_dates.txt'
eligible_dates = get_eligible_date_paths_from_file(eligible_dates_txt_output)
# print(f"Eligible dates loaded: {eligible_dates}")
import time

start_time = time.time()
daily_PnL, dates, success_rate = execute_trading_strategy(
                                    lookback_window=60,
                                    lookforward_window=3,
                                    w=5,
                                    eligible_dates=eligible_dates,
                                    num_dates = num_dates,
                                    num_med = num_med,
                                    num_clusters = num_clusters,
                                    win_threshold = win_threshold,
                                    cluster_selection = cluster_selection,
                                    num_trading_clusters = num_trading_clusters,
                                    weight_type = weight_type,
                                    winsorize_raw = winsorize_raw,
                                    winsorize_res = winsorize_res,
                                    winsor_param = winsor_param)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")

sharpe_ratio = calculate_Sharpe_Ratio(daily_PnL)
cumulative_pnl = np.cumsum(daily_PnL)

# Save the plot in a 'results' folder in the same directory as this .py file
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for Jupyter or interactive sessions
    script_dir = os.getcwd()
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, 'result.csv')
# Append daily_PnL as a new row, with dates as header only if file is empty
write_header = not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0
with open(csv_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    param_values = [
        num_dates,
        num_med,
        num_clusters,
        win_threshold,
        cluster_selection,
        num_trading_clusters,
        weight_type,
        winsorize_raw,
        winsorize_res,
        winsor_param,
        sharpe_ratio,
        success_rate
    ]
    if write_header:
        header = [
            'num_dates',
            'num_med',
            'num_clusters',
            'win_threshold',
            'cluster_selection',
            'num_trading_clusters',
            'weight_type',
            'winsorize_raw',
            'winsorize_res',
            'winsor_param',
            'sharpe_ratio',
            'success_rate'
        ] + [str(date) for date in dates]
        writer.writerow(header)
    writer.writerow(param_values + list(daily_PnL))
print(f"PnL values appended to {csv_path} (dates as columns, each run as a new row)")

