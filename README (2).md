# Neural-Network Virtual Sensors for Engine Performance and Emissions

A Python/PyTorch project investigating **data-driven virtual sensors** for predicting **nitrogen oxide (NOx) emissions** and **fuel consumption** from engine operating measurements. The project combines exploratory data analysis, data cleaning, feature engineering, feed-forward neural networks, hyperparameter sweeps, and SHAP-based interpretation.

The implementation was developed as an academic engineering assignment.

## Project overview

The aim is to approximate relationships between measured engine operating variables and quantities of interest that would otherwise require dedicated measurement or modelling. Two regression targets are trained separately:

- **NOx concentration** (reported in ppm in the code's plot labels).
- **Fuel consumption** (reported in kg/h in the code's plot labels).

The models are conventional **fully connected multilayer perceptrons (MLPs)**, not recurrent or time-series models. The `Time` field is not used as an input feature.

## Dataset Availability

The project uses an engine operational dataset
(Data2026.csv), which is not included in this repository.

The source code, preprocessing methodology, model
architectures, and evaluation procedures are provided
for reference.

Running the full pipeline requires the original dataset.

## Workflow

```text
Tab-separated engine dataset (Data2026.csv)
             |
             v
Initial exploration and data cleaning
             |
             v
RPM-grouped train / validation / test split
             |
             v
Standardisation using training-set statistics
             |
             v
PyTorch feed-forward neural networks
             |
             v
Weights & Biases hyperparameter sweeps
             |
             v
Prediction metrics and diagnostic plots
             |
             v
SHAP analysis for selected final models
```

### 1. Data preparation

The script loads `Data2026.csv` as a **tab-separated** file, skips its second row (which is expected to contain units), and removes a trailing `Unnamed: 11` column. It then:

- Removes rows containing non-positive values in the measured columns and drops duplicate records.
- Restricts `Torque Reference` and `EGR Command` to values no greater than 100 (treated as percentages).
- Applies a sequential upper-outlier filter to each measured variable, removing rows with values above **three times that variable's current mean**.
- Groups records by rounded rotational-speed bins (`rpm_group`, in increments of 100 RPM).

### 2. Input features

The baseline feature set is assembled from these columns:

| Baseline input | Description |
| --- | --- |
| `Rot. Speed` | Engine rotational speed |
| `Engine Torque` | Engine torque |
| `EGR Command` | Exhaust gas recirculation command |
| `Exhaust Gas Temperature` | Exhaust gas temperature |
| `Exhaust Gas Mass Flow` | Exhaust gas mass flow |

The code subsequently creates four interaction features:

| Engineered feature | Definition in the script |
| --- | --- |
| `Power` | `Engine Torque * Rot. Speed` |
| `Thermal_Load` | `Engine Torque * Exhaust Gas Temperature` |
| `EGR_Torque` | `Engine Torque * EGR Command` |
| `EGMF_Torque` | `Engine Torque * Exhaust Gas Mass Flow` |

These are **feature-engineering proxies**, not independently validated physical variables. In particular, the `Power` expression lacks the RPM-to-rad/s conversion required for mechanical power in watts.

### 3. Model architecture and training

A configurable PyTorch MLP (`NNModel`) uses linear layers and **ReLU** activations, with one scalar regression output. Training uses:

- **Mean squared error (MSE)** as the loss function.
- The **Adam** optimizer.
- Input and target **`StandardScaler`** objects fitted exclusively on the training partition.
- A fixed random seed (`42`) for NumPy and PyTorch.
- Full-batch optimisation for a configurable number of epochs, with training and validation losses tracked separately.

Records are split **by groups of rotational speed**, rather than by random individual rows: approximately 70% of *RPM groups* for training, 10% for validation, and 20% for testing. These are proportions of groups, **not necessarily of data samples**.

### 4. Hyperparameter experiments

The code uses [Weights & Biases](https://wandb.ai/) (W&B) grid sweeps to compare network sizes, learning rates, and training durations.

| Configuration | Hidden-layer sizes | Learning rates | Epochs |
| --- | --- | --- | --- |
| Simple | `[16]`, `[20]` | `0.001`, `0.01` | `500`, `1000` |
| Complex | `[32]`, `[32, 32]`, `[64, 64]` | `0.001`, `0.01` | `200`, `500` |

The script runs sweeps for both outputs using baseline features, engineered features with simple networks, and engineered features with complex networks. It then runs selected **two-hidden-layer `[64, 64]` models** with SHAP enabled:

- Fuel consumption: learning rate `0.01`, `500` epochs.
- NOx: learning rate `0.001`, `500` epochs.

**Evaluation caution:** In the provided script, W&B sweep configurations minimise `Test_RMSE`. Using test performance for hyperparameter selection can produce optimistic test estimates. A stricter research workflow should select hyperparameters from **validation** performance and use the held-out test set only for final evaluation.

### 5. Evaluation and interpretation

For each training run, the code produces:

- Training and validation MSE curves.
- Predicted-versus-actual scatter plots on scaled targets.
- Predicted-versus-actual plots converted back to original target units.
- Test RMSE, coefficient of determination (**R²**), and maximum absolute error on the scaled target (**L∞**).

For the two selected final configurations, it also performs SHAP analysis via `shap.KernelExplainer`, generating:

- A **mean absolute SHAP importance** bar chart.
- A **SHAP beeswarm** summary plot.

The SHAP calculation samples up to 200 training observations for the background and 300 test observations for explanations to limit computational cost.

## Repository contents

```text
neural-network-virtual-sensors/
├── README.md
└── hw_main_1.py
```

The script additionally expects the input file `Data2026.csv` alongside it. The dataset is **not provided with this README**. Do not publish the raw measurements unless you have permission to share them.

## Installation and execution

### Prerequisites

- Python 3
- Packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `wandb`, and `shap`
- A [Weights & Biases account](https://wandb.ai/) and internet access for the sweeps and logging
- An authorised local copy of `Data2026.csv` in the same folder as `hw_main_1.py`

Using [`uv`](https://docs.astral.sh/uv/) in the repository folder:

```bash
uv run --with numpy --with pandas --with matplotlib --with seaborn --with scikit-learn --with torch --with wandb --with shap python hw_main_1.py
```

Alternatively, install the packages in a Python environment and run `python hw_main_1.py`.

> **Note:** The script calls `wandb.login()` and launches multiple grid sweeps by default. Authenticate through the normal W&B login flow; do not put API keys or private data in the repository. A full run can take substantial time, and SHAP analysis can be computationally expensive.

## Outputs

The script writes several artifacts to the **current working directory**, including:

- `results.csv` — tab-delimited summary of model configuration and evaluation metrics (despite its `.csv` suffix).
- Training/validation loss plots (PNG).
- Predicted-versus-actual plots in scaled and original units (PNG).
- SHAP bar and beeswarm plots for the selected final models (PNG).

It also creates `data.csv` and `data_sorted.csv` beside the Python file as processed versions of the source data. These processed files can still contain sensitive underlying measurements and should **not** be committed publicly without permission.

### Selected results

Numerical performance values and figures are intentionally omitted here because they **cannot be verified from the source code alone**. Once an authorised run has been completed, representative plots and a small table of held-out evaluation metrics can be added to this section.

## Scope and limitations

- The project implements a supervised-regression workflow for **NOx and fuel consumption**, not a full engine physics model.
- All experiments use a tabular, feed-forward MLP; temporal dynamics are not modelled explicitly.
- Grouping by RPM reduces sharing of identical speed groups across partitions, but does not alone prove generalisation to unseen engines or operating domains.
- The current sweep metric should be changed from test RMSE to **validation RMSE** for unbiased final testing.
- The script is a single academic workflow rather than a packaged software library; execution requires the input data and W&B access.

## Technologies

**Python · PyTorch · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Weights & Biases · SHAP**
