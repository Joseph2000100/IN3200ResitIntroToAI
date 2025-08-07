# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Create binary classification target
def create_binary_target(df):

    # Create binary target: 7+ is classed as high quality
    df['high_quality'] = (df['quality'] >= 7).astype(int)
    
    # Drop the original 'quality' column
    df = df.drop(columns='quality')
    
    return df

# Select features(X) and target (y)
def select_features_target(df):

    X = df.drop(columns='high_quality')

    y = df['high_quality']
    
    return X, y

# Split and standardize the dataset
def split_and_standardize_dataset(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# Train SVM classifier
def train_svm_classifier(X_train_scaled, y_train):
    # Initialize SVM classifier with default parameters
    svm_clf = SVC(random_state=42)
    
    # Train the classifier
    svm_clf.fit(X_train_scaled, y_train)
    
    print("\nSVM Classifier trained successfully")
    
    return svm_clf

# Evaluate the model
def evaluate_model(model, X_test_scaled, y_test, show_plot=False):
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Create a confusion matrix
    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Low Quality', 'High Quality'])
        plt.yticks([0, 1], ['Low Quality', 'High Quality'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), 
                         horizontalalignment="center", 
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.tight_layout()
        plt.show()
    
    return accuracy, cm

# Tune hyperparameters
def tune_hyperparameters(X_train_scaled, y_train):
    # Parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly']
    }
    
    # Initialize SVM classifier
    svm = SVC(random_state=42)
    
    # GridSearch
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the grid search to the data
    print("\nPerforming grid search for hyperparameter tuning...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Print best parameters and score
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_


# Load the dataset
def load_dataset(filepath):

    # Load the dataset
    try:
        # The dataset uses semicolons as separators
        df = pd.read_csv(filepath, sep=';')
        return df
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        return None

# Main script
def main():

    filepath = 'winequality-red.csv'
    df = load_dataset(filepath)
    
    if df is not None:
        # Display useful information
        print("\nDataset Information:")
        print(f"Shape: {df.shape}")
        print("\nColumn Information:")
        print(df.info())
        
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        print("\nSummary statistics:")
        print(df.describe())

        print("\nCreating Binary Classification Target")
        df = create_binary_target(df)
        print("\nDataset after creating binary target:")
        print(df.head())
        print(f"\nShape after creating binary target: {df.shape}")
        
        # Count the distribution of high quality vs low quality wines
        quality_counts = df['high_quality'].value_counts()
        print("\nDistribution of wine quality:")
        print(f"Low quality wines (0): {quality_counts.get(0, 0)}")
        print(f"High quality wines (1): {quality_counts.get(1, 0)}")
        print(f"Percentage of high quality wines: {quality_counts.get(1, 0) / len(df) * 100:.2f}%")
        
        # Select Features and Target
        print("\nSelect Features and Target")
        X, y = select_features_target(df)
        print("\nFeatures (X):")
        print(X.head())
        print("\nTarget (y):")
        print(y.head())
        
        # Split and Standardize the Dataset
        print("\nSplit and Standardize the Dataset")
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_standardize_dataset(X, y)
        
        # Display information about the split
        print(f"\nTrain set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        
        # Display class distribution in train and test sets
        print(f"\nClass distribution in training set:")
        print(f"Low quality wines (0): {(y_train == 0).sum()}")
        print(f"High quality wines (1): {(y_train == 1).sum()}")
        
        print(f"\nClass distribution in test set:")
        print(f"Low quality wines (0): {(y_test == 0).sum()}")
        print(f"High quality wines (1): {(y_test == 1).sum()}")
        
        # Train SVM Classifier with default parameters
        print("\nTraining SVM Classifier with default parameters")
        svm_model = train_svm_classifier(X_train_scaled, y_train)
        
        # Evaluate the Model with default parameters
        print("\nEvaluating the Model with default parameters")
        accuracy, confusion_mat = evaluate_model(svm_model, X_test_scaled, y_test, show_plot=False)
        
        # Tune Hyperparameters
        print("\nTuning Hyperparameters")
        tuned_model = tune_hyperparameters(X_train_scaled, y_train)
        
        # Evaluate the tuned Model
        print("\nEvaluating the tuned Model")
        tuned_accuracy, tuned_confusion_mat = evaluate_model(tuned_model, X_test_scaled, y_test, show_plot=False)
        
        # Compare models
        print("\nModel Comparison:")
        print(f"Default SVM Accuracy: {accuracy:.4f}")
        print(f"Tuned SVM Accuracy: {tuned_accuracy:.4f}")
        print(f"Improvement: {(tuned_accuracy - accuracy) * 100:.2f}%")

    else:
        print("Failed to load the dataset")

if __name__ == "__main__":
    main()