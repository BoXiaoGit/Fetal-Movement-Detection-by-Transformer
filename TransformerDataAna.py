import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import os
import random

# Specify the folder path containing the .pt files
folder_path = 'DataExtractSaved'

# Initialize a dictionary to store the loaded tensors (optional)
tensors = {}

# List all .pt files in the folder
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

# Load each .pt file one by one
for file_name in pt_files:
    file_path = os.path.join(folder_path, file_name)
    # Load the tensor
    tensor = torch.load(file_path)
    # You can process the tensor here if needed
    # Store the tensor in the dictionary with the file name as the key
    tensors[file_name] = tensor

# Initialize lists to hold the extracted data and labels
training_data = []
labels = []

# Iterate through the tensors dictionary
for file_name, data_dict in tensors.items():
    # Extract 'all_data' and 'all_labels' from each tensor's dictionary
    data_sequence = data_dict["all_data"]
    labels_sequence = data_dict["all_labels"]

    # Append the data and label sequences to their respective lists
    training_data.extend(data_sequence)  # Use extend() if the sequences are lists or arrays
    labels.extend(labels_sequence)

training_data_modified = [sample[:, :-1] for sample in training_data]
training_data = training_data_modified


def safe_normalize(data, labels):
    normalized_data = []
    normalized_labels = []

    for sample, label in zip(data, labels):
        # Ensure the sample is not empty
        if sample.size > 0:
            mean = np.mean(sample, axis=0)
            std = np.std(sample, axis=0) + 1e-10  # Avoid division by zero
            # Normalize the sample
            norm_sample = (sample - mean) / std
            normalized_data.append(norm_sample)
            normalized_labels.append(label)  # Keep the label for this valid sample
        else:
            # Optionally, handle the case for empty samples, e.g., logging or skipping
            print("Encountered an empty sample, skipping.")

    # Return lists of numpy arrays instead of converting to a single numpy array
    return normalized_data, normalized_labels


training_data_normalized, normalized_labels = safe_normalize(training_data, labels)

count_0_1 = 0
count_1_0 = 0

for label in normalized_labels:
    # Convert list to numpy array for comparison, if not already an array
    label_array = np.array(label)
    if np.array_equal(label_array, np.array([0, 1])):
        count_0_1 += 1
    elif np.array_equal(label_array, np.array([1, 0])):
        count_1_0 += 1

augment_times = 2

second_class_samples = [sample for sample, label in zip(training_data_normalized, normalized_labels) if
                        np.array_equal(label, np.array([0, 1]))]


def jittering(data, num_augmented_samples, std_dev=0.01):
    augmented_data = []
    for sample in data:
        for _ in range(num_augmented_samples):
            # Generate Gaussian noise
            noise = np.random.normal(loc=0.0, scale=std_dev, size=sample.shape)
            # Add noise to the sample
            augmented_sample = sample + noise
            augmented_data.append(augmented_sample)
    return augmented_data


augmented_startled_data = jittering(second_class_samples, augment_times, std_dev=0.1)

augmented_data_length = len(augmented_startled_data)  # For example
second_data_length = len(second_class_samples)
# Generate the labels list for the augmented data

second_class_label = [np.array([0, 1], dtype=float) for _ in range(second_data_length)]

augmented_label = [np.array([0, 1], dtype=float) for _ in range(augmented_data_length)]

augmented_data = second_class_samples + augmented_startled_data
augmented_labels = second_class_label + augmented_label

training_data_augmented = training_data_normalized + augmented_data
labels_augmented = normalized_labels + augmented_labels


def downsample_sequences(sequence_list, target_length):
    def downsample_sequence(sequence, target_length):
        original_length, num_features = sequence.shape
        # Calculate the pooling size (window size for averaging)
        pool_size = original_length // target_length

        downsampled_sequence = np.zeros((target_length, num_features))

        for i in range(target_length):
            start_index = i * pool_size
            end_index = start_index + pool_size
            # Compute the average for each feature within the window
            downsampled_sequence[i, :] = np.mean(sequence[start_index:end_index, :], axis=0)

        return downsampled_sequence

    downsampled_sequences = [downsample_sequence(sequence, target_length) for sequence in sequence_list]

    return downsampled_sequences


target_length = 200
training_data_augmented_sampled = downsample_sequences(training_data_augmented, target_length)

X = training_data_augmented_sampled
y = labels_augmented

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class TimeSeriesDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        # Convert to torch tensors
        feature = torch.tensor(feature, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)  # Assuming labels are integer values
        return feature, label


train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Create DataLoader instances to efficiently load data
batch_size = 64  # Define your batch size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerForClassification(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_classes, num_layers, dropout=0.1):
        super(TransformerForClassification, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, model_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_linear = nn.Linear(model_dim, num_classes)

    def forward(self, src):
        src = self.input_linear(src)  # [Batch size, Seq len, Features] -> [Batch size, Seq len, Model dim]
        src = src.permute(1, 0, 2)  # Transformer expects [Seq len, Batch size, Model dim]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Average over sequence
        output = self.output_linear(output)
        return output


# Model hyperparameters
input_dim = 6  # Assuming each time step has 7 features
model_dim = 512  # Size of the Transformer model
num_heads = 8  # Number of heads in multi-head attention mechanism
num_classes = 2  # Binary classification
num_encoder_layers = 2  # Number of Transformer encoder layers
dropout = 0.1  # Dropout rate

model = TransformerForClassification(input_dim, model_dim, num_heads, num_classes, num_encoder_layers, dropout)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
criterion = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(1, 6):  # 10 epochs
    for inputs, labels in train_loader:
        labels = labels.float()  # Ensure labels are float
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')
    scheduler.step()


test_output_conf = []
test_label_conf = []
for test_inputs, test_label in test_loader:
    aa = test_inputs
    # test_inputs = test_inputs.permute(1, 0, 2)

    temp = model(test_inputs)
    _, temp_output = torch.max(temp.data, 1)
    test_output_conf = test_output_conf + temp_output.numpy().tolist()
    temp_label = np.where(test_label == 1)[1]
    temp_label = temp_label.tolist()
    test_label_conf = test_label_conf + temp_label

cf_matrix = confusion_matrix(test_label_conf, test_output_conf)
TN, FP, FN, TP = cf_matrix.ravel()