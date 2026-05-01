# Bank Term Deposit Prediction — ADA442 Final Project

## Overview
Predict whether a bank client will subscribe to a term deposit using the
[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
(4,119 samples, 20 features, Portuguese banking institution 2008–2013).

## Project Structure
```
├── project.ipynb        # Jupyter Notebook (full ML pipeline)
├── app.py               # Streamlit web application
├── train_model.py       # Standalone model training script
├── bank_model.pkl       # Trained model pipeline (generated)
├── bank-additional.csv  # Dataset
├── requirements.txt     # Python dependencies
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python train_model.py        # generates bank_model.pkl
streamlit run app.py         # opens at http://localhost:8501
```

## Features
- **Data Cleaning:** Dropped `duration` (data leakage), kept `unknown` as a category
- **Feature Engineering:** `real_interest_rate` via Fisher equation
- **Model Comparison:** Logistic Regression, Random Forest, MLP Neural Network
- **Hyperparameter Tuning:** GridSearchCV with 5-fold CV (F1 scoring)
- **Deployment:** Interactive Streamlit dashboard

## Dataset
- **Source:** UCI Machine Learning Repository — Bank Marketing
- **Samples:** 4,119 (10% sample)
- **Target:** Term deposit subscription (yes / no)
