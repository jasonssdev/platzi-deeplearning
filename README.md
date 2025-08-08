# Platzi Deep Learning – Telco Churn

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
![Python >=3.11](https://img.shields.io/badge/Python-%3E%3D3.11-blue)
![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras 3.x](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

A minimal deep learning project to predict customer churn using the Telco Customer Churn dataset. Includes data preprocessing, a simple Keras Sequential model, and evaluation (confusion matrix, classification report, ROC).

## Features

- Reproducible data preprocessing (label encoding + min-max scaling)
- Simple Keras model for binary classification
- Visualizations: learning curves, confusion matrix, ROC
- Saved artifacts: fitted encoders and scaler
- Works on macOS CPU by default; Colab-friendly

## Project Structure

```
platzi-deeplearning/
├─ notebooks/
│  └─ 01_jss_churn.ipynb
├─ utils/
│  └─ paths.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ models/                 # serialized encoders/scalers/models
├─ README.md
└─ .gitignore
```

## Getting Started

### Prerequisites

- macOS, Python 3.11+
- pip and venv (or conda)
- Optional: Kaggle CLI for dataset download

### Setup (macOS)

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate

# upgrade tooling
python -m pip install -U pip

# install core libs
python -m pip install -U tensorflow-macos keras scikit-learn pandas seaborn matplotlib joblib ydata-profiling kaggle

# optional: developer tooling
python -m pip install -U black isort flake8 pre-commit
pre-commit install
```

Apple Silicon note:
- If you see TF-Metal GPU crashes, prefer CPU for stability during this course work. In the notebook, set:
  ```python
  import os
  os.environ["TF_METAL_ENABLED"] = "0"
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  ```
  and restart the kernel.

### Dataset

Download the Telco Customer Churn dataset with Kaggle CLI:

```bash
# configure Kaggle on macOS
mkdir -p ~/.kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json and:
chmod 600 ~/.kaggle/kaggle.json

# from repo root (adjust target folder if needed)
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

Alternatively, the notebook already contains a Kaggle download cell that uses utils/paths.py to resolve directories.

## Usage

- Open notebooks/01_jss_churn.ipynb in VS Code and run cells top to bottom.
- Artifacts:
  - models/label_encoders.pkl
  - models/scaler.pkl
  - Optionally save trained Keras models to models/.

### Run in Google Colab (optional)

- Upload this repository or the notebook to a GitHub repo.
- In Colab: File > Open Notebook > GitHub, select the notebook.
- First cell in Colab:
  ```python
  !pip -q install keras tensorflow scikit-learn pandas seaborn matplotlib joblib ydata-profiling kaggle
  ```

## Evaluation

The notebook computes:
- Accuracy over epochs (train/validation)
- Confusion matrix and classification report
- ROC curve with AUC (train/validation)

You can tweak model capacity, optimizer, learning rate, batch size, and class weights to improve recall on the minority class.

## Development

- Formatting: Black + isort
- Linting: flake8
- Git hooks: pre-commit

Common commands:

```bash
# format
black .
isort .

# lint
flake8

# run notebook from VS Code’s native UI (recommended)
```

## Reproducibility

- Set random seeds at the start of your notebook/script:
  ```python
  import os, random, numpy as np, tensorflow as tf
  SEED = 42
  os.environ["PYTHONHASHSEED"] = str(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  ```
- Log versions: TensorFlow, Keras, Python, and dataset snapshot.

## Troubleshooting

- Kernel dies on Apple Silicon with TF-Metal:
  - Force CPU: `tf.config.set_visible_devices([], "GPU")`
  - Disable Metal via env vars (see Getting Started)
  - Update packages: `python -m pip install -U tensorflow-macos tensorflow-metal keras`
  - Restart the kernel after changes

## Roadmap

- Add unit tests for preprocessing utilities
- Add model checkpointing and experiment tracking (MLflow/W&B)
- Add CI workflow (lint + basic tests)
- Containerize with Docker for portability

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files
