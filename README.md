# Random Bayesian Forests for Futures Price Prediction

Predicting futures prices using a novel approach of combining Random Forests with Bayesian updates.

## Introduction

This project aims to predict the future prices of financial assets using a hybrid model that combines the ideas of Random Forests with Bayesian statistics. Random Forests are known for their capability to handle complex datasets and avoid overfitting, while Bayesian updates provide a principled way of updating our beliefs based on new data.

## Features

- **Random Bayesian Trees**: A single decision tree with Bayesian updates at each node.
- **Random Bayesian Forest**: An ensemble of Random Bayesian Trees.
- **Rolling Window Backtesting**: Evaluate the model on different windows of data to ensure robustness.
- **Visualization Tools**: View the structure and decisions of individual trees in the forest.

## How It Works

1. **Data Preprocessing**:
    - Data is normalized and outliers are removed.
    - Features like day of the week and month are engineered for better prediction accuracy.
  
2. **Model Training**:
    - Data is split into training and test sets.
    - Random Bayesian Forests are trained on the training data.

3. **Model Evaluation**:
    - The model's predictions are compared against actual values.
    - Metrics like RMSE and MAE are used for evaluation.

4. **Visualization**:
    - Individual trees in the forest can be visualized using the provided tools.

## Dependencies

- pandas
- numpy
- scikit-learn
- yfinance
- graphviz

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/OmidVHeravi/RandomBayesianForest.git
cd RandomBayesianForest

3. Install the required packages:
```bash
pip install -r requirements.txt

3. Run the main script
```bash
python rbf.py
