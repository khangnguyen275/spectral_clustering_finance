import subprocess
import itertools
import numpy as np
import time
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

# weighting methods 
weighting_methods = ["linear", "exponential"]
weighting_cmds = []
for method in weighting_methods:
    cmd = [
        "python", "main_experiment.py",
        "--num_dates", "-1",  # Use all dates
        "--num_med", "self", 
        "--num_clusters", "40",
        "--win_threshold", "0.001",
        "--num_trading_clusters", "40",
        "--weight_type", method,  # Use current weighting method
        "--winsor_param", "0",
    ]
    weighting_cmds.append(cmd)

# winsorization methods
winsorization_combinations = [
    ("--winsorize_raw", "--winsor_param 0.05"),
    ("--winsorize_res", "--winsor_param 0.05"),
    ("--winsorize_raw", "--winsor_param 0.01"),
    ("--winsorize_res", "--winsor_param 0.01")
]

cluster_cmds = []
for winsor_type, winsor_param in winsorization_combinations:
    # Split winsor_param into flag and value
    param_flag, param_value = winsor_param.split()
    cmd = baseline_cmd.copy()
    # Remove the default --winsor_param and its value
    if "--winsor_param" in cmd:
        idx = cmd.index("--winsor_param")
        cmd.pop(idx)  # remove flag
        cmd.pop(idx)  # remove value
    # Add the new winsorization flags
    cmd.append(winsor_type)
    cmd.append(param_flag)
    cmd.append(param_value)
    cluster_cmds.append(cmd)


all_cmds = weighting_cmds + cluster_cmds
# Execute all commands
for cmd in all_cmds:
    try:
        print(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)

