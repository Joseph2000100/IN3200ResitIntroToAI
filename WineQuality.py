# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

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
        # Display some basic information
        print("\nDataset Information:")
        print(f"Shape: {df.shape}")
        print("\nColumn Information:")
        print(df.info())
        
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        print("\nSummary statistics:")
        print(df.describe())
        
    else:
        print("Failed to load the dataset")

if __name__ == "__main__":
    main()