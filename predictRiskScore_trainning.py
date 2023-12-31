import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, Binarizer, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import predictRiskScore_nnModel as net
import CONSTANTS as const


# Load your data from a CSV file or any other source
# Replace 'your_file_path.csv' with the actual path to your CSV file
file_path = 'dataset_MLtrainingVectors_fraudRiskScore_labeling.csv'
data_df = pd.read_csv(file_path, skiprows=1)

scaler = MinMaxScaler()
data_df.iloc[:, :7] = scaler.fit_transform(data_df.iloc[:, :7])

binarizer = Binarizer(threshold=0.5)
data_df.iloc[:, 7:9] = binarizer.fit_transform(data_df.iloc[:, 7:9])

# Extract features and labels
X = data_df.iloc[:, :-1].values  # Features
y = data_df.iloc[:, -1].values  # Labels
# print(X)
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Assuming integer labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, stratify=y, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Use DataLoader for efficient batch loading during training and testing
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = X_train.shape[1]
output_size = const.output_size #4  # Number of classes (0, 1, 2, 3)

model = net.NeuralNetwork(input_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
epochs = 40
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Evaluation on the test set
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy on the test set: {:.2f}%'.format(100 * correct / total))
model_path = 'trained_riskScore_model.pth'
torch.save(model.state_dict(), model_path)