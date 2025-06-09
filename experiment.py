import sys
import os
import argparse
import re
import matplotlib.pyplot as plt
from utils.metrics import *

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
parser.add_argument('--cluster_selection', action='store_true', help='Enable cluster selection (default: False)')
parser.add_argument('--weight_type', choices=['uniform', 'linear', 'exponential'], default='uniform', help='Type of weighting to use (default: uniform)')
parser.add_argument('--winsorize_raw', action='store_true', help='Enable windsorization for the raw returns (default: False)')
parser.add_argument('--winsorize_res', action='store_true', help='Enable windsorization for the res returns (default: False)')
parser.add_argument('--winsor_param', type=float, default=0.05, help='Winsorization parameter (default: 0.05)')
parser.add_argument('--num_dates', type=int, default=None, help='Number of eligible dates to use (default: None)')
args = parser.parse_args()

cluster_selection = args.cluster_selection
weight_type = args.weight_type
winsorize_raw = args.winsorize_raw
winsorize_res = args.winsorize_res
winsor_param = args.winsor_param
num_dates = args.num_dates

print(f"cluster_selection: {cluster_selection}, weight_type: {weight_type}, windsorize_raw: {winsorize_raw}, windsorize_res: {winsorize_res}, winsor_param: {winsor_param}")

eligible_dates_txt_output = path + '/eligible_dates.txt'
eligible_dates = get_eligible_date_paths_from_file(eligible_dates_txt_output)
# print(f"Eligible dates loaded: {eligible_dates}")
import time

start_time = time.time()
daily_PnL, dates = execute_trading_strategy(win_threshold=0.1,
                                     lookback_window=60,
                                     lookforward_window=3,
                                     w=5,
                                     eligible_dates=eligible_dates,
                                     cluster_selection=cluster_selection,
                                     num_dates=num_dates)
                                    #  weighting_scheme=weight_type,
                                    #  winsorize_raw=winsorize_raw,
                                    #  winsorize_res=winsorize_res,
                                    #  winsor_param=winsor_param)
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
run_setting = str(cluster_selection) + "_" + weight_type + "_" + str(winsorize_raw) + "_" + str(winsorize_res) + "_" + str(winsor_param)

plot_path = os.path.join(results_dir, f'{run_setting}_cumulative_pnl.jpg')
PnL_path = os.path.join(results_dir, f'{run_setting}_daily_PnL')
Sharpe_path = os.path.join(results_dir, f'{run_setting}_Sharpe_Ratio')
date_path = os.path.join(results_dir, f'{run_setting}_date')



np.savetxt(PnL_path + '.txt', daily_PnL)
np.savetxt(Sharpe_path + '.txt', sharpe_ratio)
np.savetxt(date_path + '.txt', dates, fmt='%s')

plt.figure(figsize=(14, 7))
plt.plot(dates, cumulative_pnl)
plt.title('Cumulative Daily PnL')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.xticks(rotation=45, fontsize=8)
plt.grid(True)
# Show only a subset of x-ticks for readability
if len(dates) > 20:
    step = max(1, len(dates) // 20)
    plt.xticks(dates[::step])
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Plot saved to {plot_path}")

