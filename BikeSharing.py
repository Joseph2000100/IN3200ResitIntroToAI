# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Calculate mean squared error
def mean_squared_error(y_true, y_pred):
   return np.mean((y_true - y_pred) ** 2)


# Calculate the R^2 coefficient
def r2_score(y_true, y_pred):
   ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
   ss_residual = np.sum((y_true - y_pred) ** 2)
   return 1 - (ss_residual / ss_total)

# Load the dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')

    # Extract features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    return df


# Preprocess data
def preprocess_data(df):
    # Extract features for the model
    features = ['temp', 'atemp', 'hum', 'windspeed']
    X = df[features].values
    y = df['cnt'].values
    
    # Split data 80:20
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, features


# Handle plotting
def create_exploratory_plots(df):

    plt.style.use('default')

    # Create a graph with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Temperature vs Demand
    axes[0, 0].scatter(df['temp'], df['cnt'], alpha=0.5)
    axes[0, 0].set_title('Temperature vs Bike Demand')
    axes[0, 0].set_xlabel('Temperature')
    axes[0, 0].set_ylabel('Number of Bikes Rented')
    axes[0, 0].grid(True, alpha=0.3)

    # Humidity vs Demand
    axes[0, 1].scatter(df['hum'], df['cnt'], alpha=0.5)
    axes[0, 1].set_title('Humidity vs Bike Demand')
    axes[0, 1].set_xlabel('Humidity')
    axes[0, 1].set_ylabel('Number of Bikes Rented')
    axes[0, 1].grid(True, alpha=0.3)

    # Wind Speed vs Demand
    axes[1, 0].scatter(df['windspeed'], df['cnt'], alpha=0.5)
    axes[1, 0].set_title('Wind Speed vs Bike Demand')
    axes[1, 0].set_xlabel('Wind Speed')
    axes[1, 0].set_ylabel('Number of Bikes Rented')
    axes[1, 0].grid(True, alpha=0.3)

    # Hour vs Demand
    axes[1, 1].bar(df['hour'], df['cnt'], alpha=0.5)
    axes[1, 1].set_title('Hour of Day vs Bike Demand')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Number of Bikes Rented')
    axes[1, 1].grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()
    plt.show()


# Linear Regression with Gradient Descent
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    # Batch gradient descent
    def fit_batch(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    # Stochastic gradient descent
    def fit_sgd(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Update for each sample
            for i in range(n_samples):
                xi = X_shuffled[i].reshape(1, n_features)
                yi = y_shuffled[i]

                y_pred = np.dot(xi, self.weights) + self.bias

                dw = np.dot(xi.T, (y_pred - yi))
                db = (y_pred - yi)

                self.weights -= self.learning_rate * dw.flatten()
                self.bias -= self.learning_rate * db

        return self

    # Prediction function
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Main script
def main():
    # Load the data
    filepath = 'BikeSharingDataset/hour.csv'
    df = load_dataset(filepath)

    # Display some basic information
    print("Dataset Information:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Create exploratory plots
    create_exploratory_plots(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, features = preprocess_data(df)
    print(f"\nFeatures used for the model: {features}")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")


if __name__ == "__main__":
    main()
