import torch
import torch.nn as nn
#import torch.optim as optim
#from sklearn.preprocessing import StandardScaler, Binarizer, MinMaxScaler
#from torch.utils.data import DataLoader, TensorDataset
#from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


"""
Define Neural Network Class.
"""

# Define the neural network model
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size , 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, output_size)  # Output layer with output_size neurons
#
#
#
#     def forward(self, x):
#
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#
#         return x


# Instantiate the model with the correct input_size and output_size

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.3):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


    