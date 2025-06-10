import subprocess
import itertools
import numpy as np
np.set_printoptions(precision=8, suppress=False)

baseline_cmd = [
    "python", "main_experiment.py",
    "--num_dates", "-1",  # Use all dates
    "--num_med", "self", 
    "--num_clusters", "40",
    "--win_threshold", "0.001",
    "--num_trading_clusters", "40",
    "--weight_type", "uniform",
    "--winsor_param", "0",
]

# cluster selection commands
cluster_combinations = [
    # (num_dates, num_med, num_clusters, win_threshold, cluster_selection_flag, num_trading_clusters, weight_type, winsor_param)
    ("-1", "self", "40", "0.001", True,  "5",  "uniform", "0"),
    ("-1", "self", "40", "0.001", True,  "10", "uniform", "0"),
    ("-1", "self", "40", "0.001", True,  "20", "uniform", "0"),
    ("-1", "self", "40", "0.001", True,  "30", "uniform", "0"),
    ("-1", "self", "5",  "0.001", False, "5",  "uniform", "0"),
    ("-1", "self", "10", "0.001", False, "10", "uniform", "0"),
    ("-1", "self", "20", "0.001", False, "20", "uniform", "0"),
    ("-1", "self", "30", "0.001", False, "30", "uniform", "0"),
]
cluster_cmds = []
for combo in cluster_combinations:
    cmd = [
        "python", "main_experiment.py",
        "--num_dates", combo[0],
        "--num_med", combo[1],
        "--num_clusters", combo[2],
        "--win_threshold", combo[3],
    ]
    if combo[4]:  # cluster_selection flag
        cmd.append("--cluster_selection")
    cmd += [
        "--num_trading_clusters", combo[5],
        "--weight_type", combo[6],
        "--winsor_param", combo[7],
    ]
    cluster_cmds.append(cmd)

all_cmds = [baseline_cmd] + cluster_cmds
# Execute all commands
for cmd in all_cmds:
    try:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)

