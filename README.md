# Math 285J Project - Ablation Study for Correlation Matrix Clustering for Statistical Arbitrage Portfolios

This repository contains code and experiments for the Math 285J: Topics in Data Science and Machine Learning for Finance final project with Professor Mihai Cucuringu. The authors for this project are Yifan Gu, Khang Nguyen, and Lunji Zhu.

> **Project description:**  
> Financial data suffers from the curse of dimensionality and is well-known to be ill-conditioned. To tackle these issues, techniques such as dimensionality reduction and clustering have previously been employed. Recently, Jin et al. (2023) [^1] and Khelifa et al. (2024) [^2] proposed constructing portfolio strategies using signed graph clustering methods such as the Signed Positive Over Negative Generalized Eigenproblem (SPONGE) [^3]. These approaches generate profitable trading strategies with over 10% annualized returns and statistically significant Sharpe ratios above one.  
>  
> In this project, we expand on this idea by conducting further ablation studies to see the effect of using different techniques to construct the signed graph-based portfolios.

[^1]: Jin, Q., Cucuringu, M., & Cartea, Á. (2023). Correlation matrix clustering for statistical arbitrage portfolios. In *Proceedings of the Fourth ACM International Conference on AI in Finance* (pp. 557–564).

[^2]: Khelifa, N., Allier, J., & Cucuringu, M. (2024). Cluster-driven Hierarchical Representation of Large Asset Universes for Optimal Portfolio Construction. In *Proceedings of the 5th ACM International Conference on AI in Finance* (pp. 177–185).

[^3]: Cucuringu, M., et al. (2019). SPONGE: Signed Positive Over Negative Generalized Eigenproblem.

See the [project report](./285J_Project_Report.pdf) and [slides](./285J_Project_Slides.pdf) in this github page for more details.

## Project Structure
- `preprocessing.py`: Preprocesses CRSP data by validating file paths and filtering dates based on the percentage of zero values in the 'pvCLCL' column. 
- `main_experiment.py`: Main script to run a single experiment.
- `run_main_experiment_khang.py` & `run_main_experiment_lunji.py`: Scripts for batch runs; they execute multiple experiments and save results as CSV files in the `/results` directory.
- `plot_creation_1.ipynb` & `plot_creation_2.ipynb`: Notebooks used to generate plots from experiment results.
- `/utils`: Directory containing most of the functions that make up the trading strategy.
- `/results`: Directory containing output CSV files from batch runs.

## Usage

To set up the environment, use Conda with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate math285j
```

### File Structure
Here's the file structure in the authors' folder
```
math285j_project/
├── environment.yml
├── main_experiment.py
├── plot_creation_1.ipynb
├── plot_creation_2.ipynb
├── preprocessing.py
├── README.md
├── results/
│   └── (output CSV files)
├── data/
│   └── (include the data set)
├── run_main_experiment_khang.py
├── run_main_experiment_lunji.py
└── utils/
    └── (utility modules and functions)
```

### Data Preprocessing
Before running the experiments, download the data from
 [this Dropbox folder](https://www.dropbox.com/scl/fo/bx9jv5x2fnu9j02i5j77a/AGljwCnU10aplUA6BZ3AwrU?rlkey=7jeselv0gnki38q4dvrzbhbsc&e=1&st=qragjz2p&dl=0) and save the yearly data to the `data` folder. Then preprocess the CRSP data by executing:

```bash
python preprocessing.py
```

This step validates file paths and filters dates based on the percentage of zero values in the 'pvCLCL' column.

### Run a Single Experiment

```bash
python main_experiment.py
```
### Command-Line Options

The `main_experiment.py` script supports several command-line arguments to customize experiment runs:

- `--num_dates`: Number of eligible dates to use (default: -1, which means all dates).
- `--num_med`: Number of medoids to use; options are `'self'` or `'var'` (default: `'self'`).
- `--num_clusters`: Number of clusters for clustering (default: 40).
- `--win_threshold`: Win threshold as a non-negative float (default: 0.001).
- `--cluster_selection`: Enable cluster selection (default: False).
- `--num_trading_clusters`: Number of trading clusters to use (default: 40).
- `--weight_type`: Type of weighting to use; options are `'uniform'`, `'linear'`, or `'exponential'` (default: `'uniform'`).
- `--winsorize_raw`: Enable winsorization for the raw returns (default: False).
- `--winsorize_res`: Enable winsorization for the residual returns (default: False).
- `--winsor_param`: Winsorization parameter (default: 0.05).

You can specify these options when running the script, for example:

```bash
python main_experiment.py --num_dates 100 --num_clusters 50 --weight_type linear
```


### Run Batch Experiments

To execute all experiments and generate results:

```bash
python run_main_experiment_khang.py
python run_main_experiment_lunji.py
```

The resulting CSV files will be saved in the `/results` folder.

## License

See `LICENSE` for details.