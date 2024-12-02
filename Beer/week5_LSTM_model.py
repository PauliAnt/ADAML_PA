import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
    
def load_data(file,seq_length):

    # Load data
    data = pd.read_csv(file,index_col=0,parse_dates=True)
    sales = data.values.reshape(-1, 1)

    # Define the sizes for train (60%), validation (20%), and test (20%) splits
    train_size = int(0.6 * len(sales))
    val_size = int(0.2 * len(sales))
    test_size = len(sales) - train_size - val_size

    # Preprocess data: normalize the sales data
    scaler = MinMaxScaler()
    scaler.fit(sales[:train_size])
    sales_scaled = scaler.transform(sales)

    # Create sequences of data
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length])
        return np.array(sequences), np.array(targets)

    # Set sequence length
    # seq_length = 12  # Using 12 months of data to predict the next month

    # Create sequences and targets
    X, y = create_sequences(sales_scaled, seq_length)

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)



    # Chronologically split the data
    train_data = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    val_data = TensorDataset(X_tensor[train_size:train_size + val_size], y_tensor[train_size:train_size + val_size])
    test_data = TensorDataset(X_tensor[train_size + val_size:], y_tensor[train_size + val_size:])

    return train_data,val_data,test_data, scaler