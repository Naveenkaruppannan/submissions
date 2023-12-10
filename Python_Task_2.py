#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


your_df = pd.read_csv('dataset-3.csv')


# In[3]:


your_df


# In[4]:


your_df.head()


# In[5]:


your_df.describe()


# In[6]:


def calculate_distance_matrix(df):
    print("Column Names:", df.columns)  # Print column names for verification
    print("Sample Data:")
    print(df.head())  # Print the first few rows for inspection

    # Create a pivot table with 'id_start' and 'id_end' as index and columns
    pivot_table = df.pivot(index='id_start', columns='id_end', values='distance')

    # Ensure the matrix is symmetric
    distance_matrix = pivot_table.add(pivot_table.T, fill_value=0)

    # Replace NaN values with zeros
    distance_matrix = distance_matrix.fillna(0)

    return distance_matrix

# Load your dataset
your_df = pd.read_csv('dataset-3.csv')

# Test the function
distance_matrix_result = calculate_distance_matrix(your_df)
distance_matrix_result


# In[7]:


def unroll_distance_matrix(distance_matrix):
    # Convert the distance matrix to a DataFrame
    df = distance_matrix.rename_axis(index='id_start').stack().reset_index(name='distance')

    # Reset column names for clarity
    df.columns = ['id_start', 'id_end', 'distance']

    # Remove rows where id_start is equal to id_end (diagonal elements)
    df = df[df['id_start'] != df['id_end']]

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    return df

# Test the function with the result from the previous question
unrolled_distance_matrix = unroll_distance_matrix(distance_matrix_result)
unrolled_distance_matrix


# In[8]:


def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Calculate the average distance for the reference value
    avg_distance = df[df['id_start'] == reference_value]['distance'].mean()

    # Calculate the lower and upper bounds of the threshold
    lower_bound = avg_distance - 0.1 * avg_distance
    upper_bound = avg_distance + 0.1 * avg_distance

    # Filter the DataFrame based on the percentage threshold
    filtered_ids = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]['id_start'].unique()

    # Sort and return the result
    return sorted(filtered_ids)

# Test the function with the unrolled distance matrix
reference_value = 1001402  # You can change this to any desired reference value
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_matrix, reference_value)
print(result_ids_within_threshold)


# In[9]:


def calculate_toll_rate(df):
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type and add columns to the DataFrame
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        result_df[vehicle_type] = result_df['distance'] * rate_coefficient

    return result_df

# Test the function with the unrolled distance matrix
result_with_toll_rates = calculate_toll_rate(unrolled_distance_matrix)
result_with_toll_rates


# In[15]:


import datetime

def calculate_time_based_toll_rates(unrolled_df):
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = unrolled_df.copy()

    # Define time ranges and discount factors for weekdays and weekends
    weekday_time_ranges = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)),
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59))
    ]

    weekend_time_range = (datetime.time(0, 0, 0), datetime.time(23, 59, 59))

    # Add your logic to calculate time-based toll rates here

    return result_df

# Test the function with the unrolled distance matrix
result_with_time_based_toll_rates = calculate_time_based_toll_rates(unrolled_distance_matrix)
print(result_with_time_based_toll_rates)


# In[ ]:




