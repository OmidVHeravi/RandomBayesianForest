"""
Random Bayesian Forest (RBF) for Time Series Prediction
-------------------------------------------------------
This module defines a probabilistic model based on decision trees.
It uses Bayesian updating within the trees and bootstrapping to form a forest.
The primary goal is to predict futures prices, but the model can be adapted for other time series data.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from graphviz import Digraph
import datetime
import yfinance as yf
from sklearn.model_selection import train_test_split


# ------------------------
# Data Structures
# ------------------------

class Data:
    """Structure to hold features and prices."""
    def __init__(self, features, prices):
        self.features = features
        self.prices = prices


class TreeNode:
    """Base class for nodes in the decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, gaussian_params=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gaussian_params = gaussian_params


class DecisionNode(TreeNode):
    """Node representing a decision based on a feature and threshold."""
    def __init__(self, feature_idx, threshold, left_child, right_child):
        super().__init__(feature_idx, threshold, left_child, right_child)


class LeafNode(TreeNode):
    """Terminal node with Gaussian parameters for Bayesian inference."""
    def __init__(self, gaussian_params):
        super().__init__(gaussian_params=gaussian_params)


# ------------------------
# Utility Functions
# ------------------------

def compute_likelihood(data, mean, variance):
    """Compute the likelihood of the data given mean and variance."""
    epsilon = 1e-8
    return np.exp(-((data.prices - mean) ** 2) / (2 * (variance + epsilon))) / np.sqrt(2 * np.pi * (variance + epsilon))


def bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance, data_size):
    """Update Gaussian parameters using Bayesian inference."""
    precision_prior = 1 / prior_variance
    precision_likelihood = data_size / likelihood_variance
    posterior_variance = 1 / (precision_prior + precision_likelihood)
    posterior_mean = (prior_mean / prior_variance + likelihood_mean * data_size / likelihood_variance) * posterior_variance
    return posterior_mean, posterior_variance


def hypothetical_split(data, feature_idx, threshold):
    """Split the data based on a feature and a threshold."""
    left_mask = data.features[:, feature_idx] < threshold
    right_mask = ~left_mask
    left_data = Data(data.features[left_mask], data.prices[left_mask])
    right_data = Data(data.features[right_mask], data.prices[right_mask])
    return left_data, right_data


def bic_or_expected_utility(data, mean, variance):
    """Compute the Bayesian Information Criterion or Expected Utility for a given node."""
    n = len(data.prices)
    log_likelihood = np.sum(np.log(compute_likelihood(data, mean, variance)))
    return log_likelihood - 1 / 2 * np.log(n)


def select_best_split(data, max_features=None):
    """Select the best feature and threshold to split the data."""
    best_score = float('-inf')
    best_split = None
    n, m = data.features.shape
    if max_features is None:
        features_to_check = range(m)
    else:
        features_to_check = np.random.choice(m, max_features, replace=False)
    for feature_idx in features_to_check:
        thresholds = np.unique(data.features[:, feature_idx])
        for threshold in thresholds:
            left_data, right_data = hypothetical_split(data, feature_idx, threshold)
            if len(left_data.prices) == 0 or len(right_data.prices) == 0:
                continue
            prior_mean, prior_variance = 0, 1
            left_likelihood_mean, left_likelihood_variance = np.mean(left_data.prices), np.var(left_data.prices)
            right_likelihood_mean, right_likelihood_variance = np.mean(right_data.prices), np.var(right_data.prices)
            left_post_mean, left_post_var = bayesian_update(prior_mean, prior_variance, left_likelihood_mean, left_likelihood_variance, len(left_data.prices))
            right_post_mean, right_post_var = bayesian_update(prior_mean, prior_variance, right_likelihood_mean, right_likelihood_variance, len(right_data.prices))
            score = bic_or_expected_utility(left_data, left_post_mean, left_post_var) + bic_or_expected_utility(right_data, right_post_mean, right_post_var)
            if score > best_score:
                best_score = score
                best_split = (feature_idx, threshold)
    return best_split

def enhanced_feature_engineering(data):
    # Previous prices (lags)
    for i in range(1, 4):  # Add 3 lags
        data[f'lag_{i}'] = data['Close'].shift(i)
    
    # Rolling statistics
    data['rolling_mean_3'] = data['Close'].rolling(window=3).mean()
    data['rolling_std_3'] = data['Close'].rolling(window=3).std()
    
    data.dropna(inplace=True)  # Drop rows with NaN values due to lag and rolling features
    return data


# ======================
# Model Training Functions
# ======================

def bayesian_decision_tree(data, max_depth, max_features=None):
    """Recursive function to train a Bayesian Decision Tree."""
    if max_depth == 0:
        likelihood_mean, likelihood_variance = np.mean(data.prices), np.var(data.prices)
        prior_mean, prior_variance = 0, 1
        post_mean, post_var = bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance, len(data.prices))
        return LeafNode((post_mean, post_var))

    best_split = select_best_split(data, max_features)
    if best_split is None:
        likelihood_mean, likelihood_variance = np.mean(data.prices), np.var(data.prices)
        prior_mean, prior_variance = 0, 1
        post_mean, post_var = bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance, len(data.prices))
        return LeafNode((post_mean, post_var))

    feature_idx, threshold = best_split
    left_data, right_data = hypothetical_split(data, feature_idx, threshold)
    left_child = bayesian_decision_tree(left_data, max_depth - 1, max_features)
    right_child = bayesian_decision_tree(right_data, max_depth - 1, max_features)
    return DecisionNode(feature_idx, threshold, left_child, right_child)


def bootstrap(data, size=None):
    """Generate a bootstrap sample of the data."""
    if size is None:
        size = len(data.prices)
    indices = np.random.choice(len(data.prices), size, replace=True)
    return Data(data.features[indices], data.prices[indices])


def random_bayesian_forest(data, n_trees, max_depth, max_features=None):
    """Train a Random Bayesian Forest."""
    trees = []
    for _ in range(n_trees):
        bootstrapped_data = bootstrap(data)
        tree = bayesian_decision_tree(bootstrapped_data, max_depth, max_features)
        trees.append(tree)
    return trees


def traverse_tree(node, sample):
    """Traverse the tree for a given sample to get a prediction."""
    if isinstance(node, LeafNode):
        mean, variance = node.gaussian_params
        return np.random.normal(mean, np.sqrt(variance))
    if sample[node.feature_idx] < node.threshold:
        return traverse_tree(node.left, sample)
    else:
        return traverse_tree(node.right, sample)
    
def train_rbf(X_train, y_train):
  
    train_data = Data(X_train, y_train)
    forest = random_bayesian_forest(train_data, n_trees=10, max_depth=3)
    return forest

# ======================
# Evaluation and Prediction Functions
# ======================

def predict_forest(forest, sample):
    """Predict using the Random Bayesian Forest."""
    predictions = [traverse_tree(tree, sample) for tree in forest]
    return np.mean(predictions)

def predict_price_for_today(data, window_size, scaler_close):
    """
    Predicts the price for today using the most recent data of size window_size.
    
    Parameters:
    - data: The processed data with features.
    - window_size: The size of the window (number of days) to use for training.
    - scaler_close: The MinMaxScaler object fitted to the 'Close' column.
    
    Returns:
    - Predicted price for today in original scale.
    """
    
    # Extract the most recent data
    latest_data = data.iloc[-window_size:]
    
    # Train the Random Bayesian Forest
    X_latest = latest_data.drop(columns=['Close']).values
    y_latest = latest_data['Close'].values
    forest = train_rbf(X_latest, y_latest)
    
    # Predict using today's features
    today_features = data.iloc[-1:].drop(columns=['Close']).values
    predicted_price_for_today = predict_forest(forest, today_features[0])
    
    # Convert to original scale
    predicted_price_for_today_original_scale = scaler_close.inverse_transform([[predicted_price_for_today]])[0][0]
    
    return predicted_price_for_today_original_scale



def evaluate_model(forest, test_data):
    """Evaluate the model using RMSE and MAE."""
    predictions = [predict_forest(forest, sample) for sample in test_data.features]
    rmse = np.sqrt(mean_squared_error(test_data.prices, predictions))
    mae = mean_absolute_error(test_data.prices, predictions)
    return rmse, mae

def rolling_window_backtest(data, window_size, model_func):
    predictions = []
    actuals = []

    for end in range(window_size + 1, len(data) + 1):
        train_data = data.iloc[end - window_size - 1:end - 1]
        test_data = data.iloc[end - 1:end]

        X_train = train_data.drop(columns=['Close']).values
        y_train = train_data['Close'].values
        X_test = test_data.drop(columns=['Close']).values
        y_test = test_data['Close'].values
        
        forest = model_func(X_train, y_train)
        
        prediction = predict_forest(forest, X_test[0])
        
        predictions.append(prediction)
        actuals.append(y_test[0])
    
    return predictions, actuals

def fetch_latest_data():
    end_date = datetime.date.today().strftime('%Y-%m-%d')  # Gets the current date
    start_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')  # Approx. 2 years ago

    data = yf.download("ES=F", start=start_date, end=end_date, interval="1d")
    data = data.dropna()

    return data


# ======================
# Visualization
# ======================

def visualize_tree(node, feature_names=None, parent_name=None, graph=None, edge_label=None):
    """Visualize a Bayesian Decision Tree using Graphviz."""
    if graph is None:
        graph = Digraph('BayesianTree', node_attr={'style': 'filled'})

    if isinstance(node, LeafNode):
        graph.node(name=str(id(node)),
                   label=f"Predict: {node.gaussian_params[0]:.2f}\nVar: {node.gaussian_params[1]:.2f}",
                   color='lightyellow')

    elif isinstance(node, DecisionNode):
        description = feature_names[node.feature_idx] if feature_names else str(node.feature_idx)
        graph.node(name=str(id(node)),
                   label=f"{description} <= {node.threshold:.2f}",
                   color='lightblue')
        
        graph.edge(str(id(node)), str(id(node.left)), label='True')
        graph.edge(str(id(node)), str(id(node.right)), label='False')
        
        visualize_tree(node.left, feature_names, str(id(node)), graph, 'True')
        visualize_tree(node.right, feature_names, str(id(node)), graph, 'False')

    return graph


if __name__ == "__main__":

    # 1. Fetch the latest data
    data = fetch_latest_data()

    # 2. Preprocess and feature engineering
    data, scaler_close = preprocess_data(data)
    data = enhanced_feature_engineering(data)

    # 3. Train-Test Split
    #X = data.drop(columns=['Close']).values
    #y = data['Close'].values
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    #train_data = Data(X_train, y_train)
    #test_data = Data(X_test, y_test)

    # 4. Model Training
    #forest = random_bayesian_forest(train_data, n_trees=10, max_depth=3)
    predictions, actuals = rolling_window_backtest(data, window_size=60, model_func=train_rbf)  # Assuming 60 days for training

    # 5. Model Evaluation
    #rmse, mae = evaluate_model(forest, test_data)
    #print(f"RMSE: {rmse}, MAE: {mae}")
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    print(f"RMSE: {rmse}, MAE: {mae}")

    # 6. Prediction for today
    #X_latest = data.iloc[-1:].drop(columns=['Close']).values
    #predicted_price = predict_forest(forest, X_latest[0])
    #predicted_price_original_scale = scaler_close.inverse_transform([[predicted_price]])[0][0]
    #print(f"Predicted price for today in original scale: {predicted_price_original_scale}")
    predicted_today = predict_price_for_today(data, window_size=60, scaler_close=scaler_close)
    print(f"Predicted price for today: {predicted_today}")



