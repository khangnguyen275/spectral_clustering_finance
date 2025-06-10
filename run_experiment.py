import subprocess
import itertools
import numpy as np
np.set_printoptions(precision=8, suppress=False)
weight_types = ["uniform"]
# winsor_params = ["0.1", "0.2"]
winsor_params = ["0"]

num_dates_options = ["-1"]  # -1 means all dates
cluster_selection_options = [True, False]
winsorize_raw_options = [False]
winsorize_res_options = [False]
# winsorize_raw_options = [True, False]
# winsorize_res_options = [True, False]

for weight_type, winsor_param, num_dates, cluster_selection, winsorize_raw, winsorize_res in itertools.product(
    weight_types, winsor_params, num_dates_options, cluster_selection_options, winsorize_raw_options, winsorize_res_options
):
    cmd = [
        "python", "experiment.py",
        "--weight_type", weight_type,
        "--winsor_param", winsor_param,
        "--num_dates", num_dates
    ]
    if winsorize_raw:
        cmd.append("--winsorize_raw")
    if winsorize_res:
        cmd.append("--winsorize_res")
    if cluster_selection:
        cmd.append("--cluster_selection")
    subprocess.run(cmd)