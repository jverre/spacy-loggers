<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-loggers: Logging utilities for spaCy

[![PyPi Version](https://img.shields.io/pypi/v/spacy-loggers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/spacy-loggers)

Starting with spaCy v3.2, alternate loggers are moved into a separate package
so that they can be added and updated independently from the core spaCy
library.

`spacy-loggers` currently provides loggers for:

- [Weights & Biases](https://www.wandb.com)
- [MLflow](https://www.mlflow.org/)
- [Comet](https://www.comet.com)

If you'd like to add a new logger or logging option, please submit a PR to this
repo!

## Setup and installation

`spacy-loggers` should be installed automatically with spaCy v3.2+, so you
usually don't need to install it separately. You can install it with `pip` or
from the conda channel `conda-forge`:

```bash
pip install spacy-loggers
```

```bash
conda install -c conda-forge spacy-loggers
```

# Loggers

## WandbLogger

### Installation

This logger requires `wandb` to be installed and configured:

```bash
pip install wandb
wandb login
```

### Usage

`spacy.WandbLogger.v4` is a logger that sends the results of each training step
to the dashboard of the [Weights & Biases](https://www.wandb.com/) tool. To use
this logger, Weights & Biases should be installed, and you should be logged in.
The logger will send the full config file to W&B, as well as various system
information such as memory utilization, network traffic, disk IO, GPU
statistics, etc. This will also include information such as your hostname and
operating system, as well as the location of your Python executable.

**Note** that by default, the full (interpolated)
[training config](https://spacy.io/usage/training#config) is sent over to the
W&B dashboard. If you prefer to **exclude certain information** such as path
names, you can list those fields in "dot notation" in the
`remove_config_values` parameter. These fields will then be removed from the
config before uploading, but will otherwise remain in the config file stored
on your local system.

### Example config

```ini
[training.logger]
@loggers = "spacy.WandbLogger.v4"
project_name = "monitor_spacy_training"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "corpus"
model_log_interval = 1000
```

| Name                   | Type            | Description                                                                                                                                                                                                                     |
| ---------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`           | The name of the project in the Weights & Biases interface. The project will be created automatically if it doesn't exist yet.                                                                                                   |
| `remove_config_values` | `List[str]`     | A list of values to exclude from the config before it is uploaded to W&B (default: `[]`).                                                                                                                                       |
| `model_log_interval`   | `Optional[int]` | Steps to wait between logging model checkpoints to the W&B dasboard (default: `None`). Added in `spacy.WandbLogger.v2`.                                                                                                         |
| `log_dataset_dir`      | `Optional[str]` | Directory containing the dataset to be logged and versioned as a W&B artifact (default: `None`). Added in `spacy.WandbLogger.v2`.                                                                                               |
| `run_name`             | `Optional[str]` | The name of the run. If you don't specify a run name, the name will be created by the `wandb` library (default: `None`). Added in `spacy.WandbLogger.v3`.                                                                       |
| `entity`               | `Optional[str]` | An entity is a username or team name where you're sending runs. If you don't specify an entity, the run will be sent to your default entity, which is usually your username (default: `None`). Added in `spacy.WandbLogger.v3`. |
| `log_best_dir`         | `Optional[str]` | Directory containing the best trained model as saved by spaCy (by default in `training/model-best`), to be logged and versioned as a W&B artifact (default: `None`). Added in `spacy.WandbLogger.v4`.                           |
| `log_latest_dir`       | `Optional[str]` | Directory containing the latest trained model as saved by spaCy (by default in `training/model-latest`), to be logged and versioned as a W&B artifact (default: `None`). Added in `spacy.WandbLogger.v4`.                       |

## MLflowLogger

### Installation

This logger requires `mlflow` to be installed and configured:

```bash
pip install mlflow
```

### Usage

`spacy.MLflowLogger.v1` is a logger that tracks the results of each training step
using the [MLflow](https://www.mlflow.org/) tool. To use
this logger, MLflow should be installed. At the beginning of each model training
operation, the logger will initialize a new MLflow run and set it as the active
run under which metrics and parameters wil be logged. The logger will then log
the entire config file as parameters of the active run. After each training step,
the following actions are performed:

- The final score is logged under the metric `score`.
- Individual component scores are logged under their default names.
- Loss values of different components are logged with the `loss_` prefix.
- If the final score is higher than the previous best score (for the current run),
  the model artifact is additionally uploaded to MLflow. This action is only performed
  if the `output_path` argument is provided during the training pipeline initialization phase.

By default, the tracking API writes data into files in a local `./mlruns` directory.

**Note** that by default, the full (interpolated)
[training config](https://spacy.io/usage/training#config) is sent over to 
MLflow. If you prefer to **exclude certain information** such as path
names, you can list those fields in "dot notation" in the
`remove_config_values` parameter. These fields will then be removed from the
config before uploading, but will otherwise remain in the config file stored
on your local system.

### Example config

```ini
[training.logger]
@loggers = "spacy.MLflowLogger.v1"
experiment_id = "1"
run_name = "with_fast_alignments"
nested = False
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
```

| Name                   | Type                       | Description                                                                                                                                                                                                             |
| ---------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_id`               | `Optional[str]`            | Unique ID of an existing MLflow run to which parameters and metrics are logged. Can be omitted if `experiment_id` and `run_id` are provided (default: `None`).                                                          |
| `experiment_id`        | `Optional[str]`            | ID of an existing experiment under which to create the current run. Only applicable when `run_id` is `None` (default: `None`).                                                                                          |
| `run_name`             | `Optional[str]`            | Name of new run. Only applicable when `run_id` is `None` (default: `None`).                                                                                                                                             |
| `nested`               | `bool`                     | Controls whether run is nested in parent run. `True` creates a nested run (default: `False`).                                                                                                                           |
| `tags`                 | `Optional[Dict[str, Any]]` | A dictionary of string keys and values to set as tags on the run. If a run is being resumed, these tags are set on the resumed run. If a new run is being created, these tags are set on the new run (default: `None`). |
| `remove_config_values` | `List[str]`                | A list of values to exclude from the config before it is uploaded to MLflow (default: `[]`).                                                                                                                            |

## CometLogger

### Installation

This logger requires `comet_ml` to be installed and configured:

```bash
pip install comet_ml
```

### Usage

`spacy.CometLogger.v1` is a logger that tracks the results of each training
steps to [Comet](https://www.comet.com) including but not limited to the full
spacy config, metrics, installed Python libraries and console logs. To use this
logger, you will need to have `comet_ml` installed and the API key configured
either through an [environment variable](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/#configure-comet-using-the-comet-config-file)
or through the [Comet config file](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/#configure-comet-using-the-comet-config-file).
Once the data is logged to Comet, you can use Comet's built-in dashboarding
capabilities to share your training runs with colleagues. If the built-in
visualization options are not enough, you can also write custom dynamic
plots in Python!

**Note** that by default, the full (interpolated)
[training config](https://spacy.io/usage/training#config) is sent over to 
MLflow. If you prefer to **exclude certain information** such as path
names, you can list those fields in "dot notation" in the
`remove_config_values` parameter. These fields will then be removed from the
config before uploading, but will otherwise remain in the config file stored
on your local system.

### Example config

```ini
[training.logger]
@loggers = "spacy.CometLogger.v1"
experiment_name = "lemmatization_impact_eval"
tags = ["lemmatixation", "evaluation"]
model_log_interval = 10
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
```

| Name                   | Type                  | Description                                                                                                                                                                                                             |
| ---------------------- | --------------------  | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_name`             | `Optional[str]`       | Name of new experiment (default: `None`).                                                                                                                                             |
| `tags`                 | `Optional[Dict[str]]` | A dictionary of strings to set as tags on the experiment (default: `None`). |
| `model_log_interval`   | `Optional[int]`       | Step interval at which to log the model to the Comet experiment (default: `None`) |
| `remove_config_values` | `List[str]`           | A list of values to exclude from the config before it is uploaded to Comet (default: `[]`).                                                                                                                            |
