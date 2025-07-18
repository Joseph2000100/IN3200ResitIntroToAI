# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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
    axes[1, 1].scatter(df['hour'], df['cnt'], alpha=0.5)
    axes[1, 1].set_title('Hour of Day vs Bike Demand')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Number of Bikes Rented')
    axes[1, 1].grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()
    plt.show()


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

    create_exploratory_plots(df)


if __name__ == "__main__":
    main()
