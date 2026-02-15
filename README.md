<p align="center">
  <img src="media/drugs-usage-models-creator-logo.png" width="700" alt="Repository logo" />
</p>
<h3 align="center">Drugs Usage Models Creator</h3>
<p align="center">Core engine for training, evaluating, and exporting specialized drug-consumption detection models<p>
<p align="center">
    <img src="https://img.shields.io/github/repo-size/lhbelfanti/drugs-usage-models-creator?label=Repo%20size" alt="Repo size" />
    <img src="https://img.shields.io/github/license/lhbelfanti/drugs-usage-models-creator?label=License" alt="License" />
</p>

---
# Drugs Usage Models Creator

## Project Purpose
This project aims to create models for the detection of "Adverse Human Behaviors" (specifically illicit drug consumption) from a Spanish Twitter corpus using various Machine Learning and Deep Learning techniques.

## Repository Structure

The project follows a structured architecture for reproducibility and maintainability:

- **/data**: Contains datasets.
  - `raw/`: Original, immutable CSV files.
  - `processed/`: Cleaned and transformed datasets ready for modeling.
- **/src**: Source code for the project.
  - `preprocessing/`: Scripts for data cleaning and preparation.
  - `features/`: Feature engineering logic.
  - `training/`: Scripts to train models.
  - `models/`: Model architecture definitions.
- **/models**: Stores trained model binaries and checkpoints.
- **/notebooks**: Jupyter notebooks for Exploratory Data Analysis (EDA) and initial experiments.
- **/evaluation**: Contains evaluation metrics, confusion matrices, and plots.

## Download this Repo

To get the code and the large model files properly:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/lhbelfanti/drugs-usage-models-creator.git
    cd drugs-usage-models-creator
    ```

2.  **Download Model Weights (Git LFS)**:
    This repository uses **Git LFS** to store large model files.
    
    *Install Git LFS if needed:*
    ```bash
    # macOS
    brew install git-lfs
    # Ubuntu
    sudo apt-get install git-lfs
    # Windows
    git lfs install
    ```

    *Pull weights:*
    ```bash
    git lfs install
    git lfs pull
    ```

## Setup Instructions

### Prerequisites
- [pyenv](https://github.com/pyenv/pyenv) with **Python 3.12.12** installed (`pyenv install 3.12.12`)

### 1. Create the virtual environment
```bash
# Use pyenv's Python 3.12.12 to create the venv
~/.pyenv/versions/3.12.12/bin/python -m venv .venv

# Activate the venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch JupyterLab
```bash
source .venv/bin/activate
jupyter lab
```
---

## License

[MIT](https://choosealicense.com/licenses/mit/)

### Logo License

Generated with AI

### Matplotlib style

The [style.mplstyle](./style.mplstyle) file was obtained from the following [repository](https://github.com/DataForScience/Networks/blob/master/d4sci.mplstyle) 