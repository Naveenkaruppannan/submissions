#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd


# In[50]:


df = pd.read_csv('dataset-1.csv')


# In[51]:


df


# In[52]:


df2 = pd.read_csv('dataset-2.csv') 


# In[53]:


df2


# In[55]:


df.describe()


# In[56]:


df.head()


# In[57]:


df.head(55)


# In[58]:


def generate_car_matrix(dataset):
    # Read the dataset
    df = pd.read_csv(dataset)

    # Create a pivot table using id_1, id_2, and car columns
    pivot_df = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    pivot_df = pivot_df.fillna(0)

    # Set diagonal values to 0
    for i in range(len(pivot_df.index)):
        pivot_df.iloc[i, i] = 0

    return pivot_df

# Replace 'dataset-1.csv' with the actual path to your CSV file
result_df = generate_car_matrix('dataset-1.csv')
result_df


# In[59]:


def get_type_count(df):
    # Add a new categorical column 'car_type'
    conditions = [
        (df['car'] <= 15),
        ((df['car'] > 15) & (df['car'] <= 25)),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=choices)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Assuming df is your DataFrame
df = pd.read_csv('dataset-1.csv')

# Call the function and print the result
result_dict = get_type_count(df)
result_dict


# In[60]:


def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Assuming df is your DataFrame
df = pd.read_csv('dataset-1.csv')

# Call the function and print the result
result_indices = get_bus_indexes(df)
print(result_indices)


# In[61]:


def filter_routes(df):
    # Group by 'route' and calculate the average value of the 'truck' column for each group
    avg_truck_by_route = df.groupby('route')['truck'].mean()

    # Filter routes where the average value of 'truck' is greater than 7
    selected_routes = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()

    return selected_routes

# Assuming df is your DataFrame
df = pd.read_csv('dataset-1.csv')

# Call the function and print the result
result_routes = filter_routes(df)
print(result_routes)


# In[62]:


def multiply_matrix(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    modified_df = input_df.copy()

    # Apply the logic: multiply values greater than 20 by 0.75, and values 20 or less by 1.25
    modified_df[modified_df > 20] *= 0.75
    modified_df[modified_df <= 20] *= 1.25

    return modified_df

# Assuming result_df is the DataFrame from Question 1
# Replace 'dataset-1.csv' with the actual path to your CSV file if necessary
result_df = generate_car_matrix('dataset-1.csv')

# Call the function and print the modified DataFrame
modified_result_df = multiply_matrix(result_df)
modified_result_df


# In[63]:


import pandas as pd
import numpy as np
from datetime import timedelta

# Set a seed for reproducibility
np.random.seed(42)

# Generate a sample dataset
num_records = 100
start_dates = pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2023-01-10'), size=num_records), format='%Y-%m-%d')
start_times = pd.to_datetime(np.random.choice(pd.date_range('00:00', '23:45', freq='15T').time, size=num_records), format='%H:%M:%S').time
end_dates = start_dates + pd.to_timedelta(np.random.randint(1, 10, size=num_records), unit='D')
end_times = pd.to_datetime(np.random.choice(pd.date_range('00:00', '23:45', freq='15T').time, size=num_records), format='%H:%M:%S').time

data = {
    'id': np.arange(1, num_records + 1),
    'id_2': np.random.choice(['A', 'B', 'C'], size=num_records),
    'startDay': start_dates.strftime('%Y-%m-%d'),
    'startTime': start_times,
    'endDay': end_dates.strftime('%Y-%m-%d'),
    'endTime': end_times
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('dataset-2.csv', index=False)

# Display the DataFrame
df


# In[64]:


def time_check(df):
    df['start_time'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_time'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    df['hour_of_day'] = df['start_time'].dt.hour
    df['day_of_week'] = df['start_time'].dt.dayofweek

    incorrect_timestamps = (
        (df['hour_of_day'] != 0) | (df['end_time'].dt.hour != 23) |
        (df['day_of_week'].nunique() != 7)
    )
    return incorrect_timestamps


# In[65]:


time_check_result = time_check(df2)


# In[66]:


# Assuming you have loaded your DataFrame, df2
time_check_result = time_check(df2)

# Display the boolean series indicating incorrect timestamps
print(time_check_result)


# In[ ]:





# In[ ]:




