# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
        
    else:
        print("Failed to load the dataset")

if __name__ == "__main__":
    main()