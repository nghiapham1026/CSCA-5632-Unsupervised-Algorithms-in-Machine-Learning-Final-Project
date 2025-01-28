import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(data, features):
    """Plot histograms and KDE plots for given features."""
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], bins=50, kde=True, color='blue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

def plot_correlation_matrix(data, filepath=None):
    """Generate a correlation matrix heatmap."""
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    if filepath:
        plt.savefig(filepath)
    plt.show()

def plot_class_distribution(data, target_column):
    """Plot the class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=data, palette="viridis")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks([0, 1], ['Legitimate (0)', 'Fraudulent (1)'])
    plt.show()
