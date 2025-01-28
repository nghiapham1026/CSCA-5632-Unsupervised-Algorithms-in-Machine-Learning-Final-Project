import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """
    Preprocess the data by handling missing values, scaling, and transforming features.
    - Log transform 'Amount'
    - Min-Max scale 'Time'
    """
    data['Log_Amount'] = np.log1p(data['Amount'])
    scaler = MinMaxScaler()
    data['Scaled_Time'] = scaler.fit_transform(data[['Time']])
    return data.drop(columns=['Amount', 'Time'])

def save_processed_data(data, filepath):
    """Save the processed dataset to a CSV file."""
    data.to_csv(filepath, index=False)
