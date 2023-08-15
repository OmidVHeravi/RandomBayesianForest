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

## Math Behind the Model

### Bayesian Update

A Bayesian update is the process of revising beliefs (probabilities) in light of new evidence. The Bayesian approach is rooted in Bayes' Theorem:

\[
P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
\]

Where:

- \( P(A|B) \) is the posterior probability of event \( A \) given evidence \( B \).
- \( P(B|A) \) is the likelihood of observing evidence \( B \) given \( A \) is true.
- \( P(A) \) is the prior probability of \( A \) (i.e., our belief about \( A \) before seeing evidence \( B \)).
- \( P(B) \) is the total probability of observing evidence \( B \).

In our model, the Bayesian update is specifically used for refining the estimates of the mean and variance for the node values in our Bayesian decision tree. The formulas for this are:

\[
\text{posterior variance} = \left( \frac{1}{\text{prior variance}} + \frac{\text{data size}}{\text{likelihood variance}} \right)^{-1}
\]

\[
\text{posterior mean} = \left( \frac{\text{prior mean}}{\text{prior variance}} + \frac{\text{likelihood mean} \times \text{data size}}{\text{likelihood variance}} \right) \times \text{posterior variance}
\]

Where the prior mean and variance represent our initial beliefs, and the likelihood mean and variance are derived from the data.

### Likelihood Calculation

The likelihood function measures the goodness of fit of our model to the data. The likelihood for our model is based on the Gaussian (or normal) distribution. The probability density function (pdf) of the Gaussian distribution is:

\[
f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
\]

Where:

- \( x \) is a data point.
- \( \mu \) is the mean of the distribution.
- \( \sigma^2 \) is the variance.

For our model, the likelihood of the data given the mean and variance is computed using this formula.

### BIC (Bayesian Information Criterion)

BIC is a criterion for model selection. It balances the likelihood of the model against its complexity (number of parameters). The formula for BIC is:

\[
BIC = -2\ln(\text{likelihood}) + k\ln(n)
\]

Where:

- \( \ln \) is the natural logarithm.
- \( k \) is the number of parameters in the model.
- \( n \) is the number of data points.

The model with the lowest BIC is generally preferred because it strikes the best balance between fit and complexity. In the context of our Bayesian decision tree, BIC helps determine the best feature and threshold for splitting the data at each node.

Certainly! Let's delve into the math and concepts behind `select_best_split` and `bayesian_decision_tree`.

---

### Selecting the Best Split (`select_best_split`)

The goal of the `select_best_split` function is to determine the optimal feature and threshold to split the data in a decision tree node.

#### Steps:

1. **Feature Selection**: 
   - If `max_features` is specified, a random subset of features is selected for consideration. Otherwise, all features are considered.
  
2. **Thresholds Consideration**:
   - For each selected feature, unique values are treated as potential thresholds.

3. **Hypothetical Split**:
   - The dataset is hypothetically split based on each feature-threshold pair.
   
4. **Bayesian Update**:
   - For the left and right subsets resulting from the split:
     - Compute the likelihood mean and variance.
     - Perform a Bayesian update to get the posterior mean and variance.
   
5. **BIC Calculation**:
   - Compute the Bayesian Information Criterion (BIC) for both left and right subsets.
   - The combined BIC score is the sum of the BIC scores for the left and right subsets.
   
6. **Optimal Split Decision**:
   - The feature-threshold pair that results in the highest combined BIC score is chosen as the optimal split.
   
The mathematical legitimacy of this approach comes from the Bayesian Information Criterion (BIC), which balances model fit (likelihood) against model complexity. By maximizing BIC, the algorithm aims to find splits that are both statistically significant and parsimonious.

### Bayesian Decision Tree (`bayesian_decision_tree`)

A Bayesian Decision Tree is a probabilistic extension of the traditional decision tree. Instead of making deterministic decisions at each node, it computes posterior distributions based on the data.

#### Steps:

1. **Termination Condition**:
   - If the maximum depth is reached or no suitable split is found, a `LeafNode` is returned with the posterior mean and variance.
   
2. **Best Split Selection**:
   - Use the `select_best_split` function to determine the optimal feature and threshold for splitting the data.
   
3. **Recursive Tree Building**:
   - Recursively build the left and right child trees using the `bayesian_decision_tree` function, decreasing the `max_depth` by one.
   
4. **Decision Node Creation**:
   - A `DecisionNode` is returned with the chosen feature, threshold, and the left and right child trees.
   
The inspiration behind Bayesian Decision Trees comes from Bayesian statistics. The idea is to use prior knowledge (prior distributions) and observed data (likelihood) to make informed decisions (posterior distributions). This approach provides a measure of uncertainty in the predictions, which can be valuable in many applications.

The legitimacy of Bayesian Decision Trees lies in the Bayesian framework's robustness. Bayesian methods offer a principled way to incorporate prior knowledge, update beliefs based on data, and quantify uncertainty. By combining the structure of decision trees with Bayesian statistics, Bayesian Decision Trees offer a powerful tool for data-driven decision-making with inherent uncertainty quantification.

These methods' mathematical and conceptual foundations are deeply rooted in statistical learning and Bayesian statistics. The fusion of decision trees with Bayesian methods offers a more nuanced and probabilistic way to make decisions based on data.

## Quick Start

```bash
1. Clone the repository:

git clone https://github.com/OmidVHeravi/RandomBayesianForest.git
cd RandomBayesianForest

2. Install the required packages:

pip install -r requirements.txt

3. Run the main script

python rbf.py

