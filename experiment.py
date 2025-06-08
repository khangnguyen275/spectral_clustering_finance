import sys
import os
possible_paths = [
        '/Users/khang/Desktop/math285j_project/data/drive-download-20250531T145738Z-1-001/CRSP Data Set',
        '/Users/lunjizhu/Desktop/MATH 285J Project Workspace/spectral_clustering_finance/data',
        'F:/spectral_clustering_finance/data/drive-download-20250531T145738Z-1-001/CRSP Data Set',
        '/Users/yifangu/Desktop/MATH 285J/285J Project/spectral_clustering_finance/data'
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

eligible_dates_txt_output = path + '/eligible_dates.txt'
eligible_dates = get_eligible_date_paths_from_file(eligible_dates_txt_output)
print(f"Eligible dates loaded: {eligible_dates}")
import time

start_time = time.time()
daily_PnL = execute_trading_strategy(win_threshold=0.1,
                                     lookback_window=60,
                                     lookforward_window=3,
                                     w=5,
                                     eligible_dates=eligible_dates)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")

import matplotlib.pyplot as plt

# Get the dates from eligible_dates (assume each path ends with YYYYMMDD.csv.gz)
import re
date_labels = []
date_pattern = re.compile(r'(\\d{8})\\.csv\\.gz$')
for path in eligible_dates[:len(daily_PnL)]:
    match = re.search(r'(\\d{8})\\.csv\\.gz$', path)
    if match:
        date_labels.append(match.group(1))
    else:
        date_labels.append("")

cumulative_pnl = np.cumsum(daily_PnL)


# Save the plot in a 'results' folder in the same directory as this .py file
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
plot_path = os.path.join(results_dir, 'cumulative_pnl.png')


plt.figure(figsize=(14, 7))
plt.plot(date_labels, cumulative_pnl)
plt.title('Cumulative Daily PnL')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.xticks(rotation=45, fontsize=8)
plt.grid(True)
# Show only a subset of x-ticks for readability
if len(date_labels) > 20:
    step = max(1, len(date_labels) // 20)
    plt.xticks(date_labels[::step])
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Plot saved to {plot_path}")

