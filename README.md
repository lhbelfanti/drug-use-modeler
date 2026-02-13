<p align="center">
  <img src="media/drugs-usage-models-creator-logo.png" width="300" alt="Repository logo" />
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

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
