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

# Base directory where the files are located
base_dir = 'DataSaved/NO59 patient/'

# Paths to your files
path_original_csv = base_dir + '1_US_059SZ.csv'
path_annotation_xlsx = base_dir + '1_US_059SZ_annotation_20240308120413.xlsx'  # Excel file

# Reading the CSV files into pandas DataFrames
df_original = pd.read_csv(path_original_csv, encoding='ISO-8859-1')

# Reading the Excel file into a pandas DataFrame
df_annotation_info = pd.read_excel(path_annotation_xlsx, usecols=[0, 4], skiprows=[0],
                                   names=['Time_in_s', 'Movement_Type'])

# Displaying the first few rows of each DataFrame to confirm successful loading
print("First few rows of the annotated data file:")
print(df_original.head())

print("\nFirst few rows of the additional annotation file (Excel):")
print(df_annotation_info.head())

# Count occurrences of each movement type in the 'Movement_Type' column
movement_counts = df_annotation_info['Movement_Type'].value_counts()

# Display counts for "General" and "Startled"
print("Count for 'General':", movement_counts.get("General", 0))
print("Count for 'Startled':", movement_counts.get("Startled", 0))

general_times = df_annotation_info[df_annotation_info['Movement_Type'] == 'General']['Time_in_s']

# Filter the DataFrame for 'Startled' movement type and select the 'Time_in_s' column
startled_times = df_annotation_info[df_annotation_info['Movement_Type'] == 'Startled']['Time_in_s']

# Display the times
print("Times for 'General' movements:")
print(general_times.to_string(index=False))  # .to_string(index=False) for a cleaner display without the index

print("\nTimes for 'Startled' movements:")
print(startled_times.to_string(index=False))


# Define a function to create intervals around each timing
def create_intervals(time_series):
    return time_series.apply(lambda x: [x - 2, x + 3])


# Apply the function to the 'General' and 'Startled' times
general_intervals = create_intervals(general_times)
startled_intervals = create_intervals(startled_times)

# Display the intervals
print("Intervals for 'General' movements:")
print(general_intervals.to_string(index=False))

print("\nIntervals for 'Startled' movements:")
print(startled_intervals.to_string(index=False))

sampling_rate = 512  # Data sampling rate in Hz


# Function to convert intervals to sample indices, retrieve corresponding data, and select specific columns
def get_sampled_data(intervals, df_data):
    sampled_data_frames = []

    for interval in intervals:
        # Convert time interval to sample index range
        start_index = int(round(interval[0] * sampling_rate))
        end_index = int(round(interval[1] * sampling_rate))

        # Slice the DataFrame for the current interval and specific columns (2nd to 7th)
        sampled_data_frame = df_data.iloc[start_index:end_index, 1:8]  # Adjusted to select columns 3rd to 8th

        # Add the sliced DataFrame to the list
        sampled_data_frames.append(sampled_data_frame)

    return sampled_data_frames


# Assuming 'general_intervals' is a list of intervals for "General" movements
# and 'df_data' is your DataFrame containing the sampled data
general_intervals_sampled = get_sampled_data(general_intervals, df_original)
startled_intervals_sampled = get_sampled_data(startled_intervals, df_original)

# Example of how to access the first sampled data sequence for 'General' movements
if general_intervals_sampled:  # Check if there's at least one interval
    print("Sampled data sequence for the first 'General' interval (columns 3rd to 8th):")
    print(general_intervals_sampled[0])

# Assuming 'startled_intervals' is your list of intervals for "Startled" movements
# and 'df_data' is your DataFrame containing the sampled data


# Example of how to access the first sampled data sequence for 'Startled' movements
if startled_intervals_sampled:  # Check if there's at least one interval
    print("Sampled data sequence for the first 'Startled' interval (columns 3rd to 8th):")
    print(startled_intervals_sampled[0])

general_data_numpy_list = [df.to_numpy() for df in general_intervals_sampled]
startled_data_numpy_list = [df.to_numpy() for df in startled_intervals_sampled]

general_labels = [torch.tensor([1, 0]).float().numpy().copy()] * len(general_data_numpy_list)
startled_labels = [torch.tensor([0, 1]).float().numpy().copy()] * len(startled_data_numpy_list)

data_sequence = general_data_numpy_list + startled_data_numpy_list
labels = general_labels + startled_labels

data_dict = {
    "all_data": data_sequence,
    "all_labels": labels
}
torch.save(data_dict, 'DataExtractSaved/data_dict_59_1.pt')
