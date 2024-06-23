#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from tensorflow.keras.models import load_model
import pickle
from scipy.optimize import linprog



# In[50]:


from google.colab import drive
drive.mount('/content/drive')


# In[51]:


#/content/drive/MyDrive/Data/HomeC.csv

smart_home_lstm = pd.read_csv('/content/drive/MyDrive/Data/smart_home_data_EDA.csv')
#  Drop the 'index' column
smart_home_lstm = smart_home_lstm.drop(columns=['index'])


# In[59]:


# Check if 'time' is already set as index
if 'time' not in smart_home_lstm.index.names:
    smart_home_lstm['time'] = pd.to_datetime(smart_home_lstm['time']) # Convert 'time' column to datetime object
    smart_home_lstm.set_index('time', inplace=True)
# Select only numeric columns for resampling
numeric_columns = smart_home_lstm.select_dtypes(include=np.number).columns
smart_home_lstm = smart_home_lstm[numeric_columns].resample('H').mean()

# Encode categorical features
smart_home_lstm['weekday'] = LabelEncoder().fit_transform(pd.Series(smart_home_lstm.index).apply(lambda x: x.day_name())).astype(np.int8)
smart_home_lstm['timing'] = LabelEncoder().fit_transform(smart_home_lstm['hour'].apply(lambda x: 'Morning' if 5 <= x < 12 else 'Afternoon' if 12 <= x < 18 else 'Evening' if 18 <= x < 22 else 'Night')).astype(np.int8)


# Select features and target variable
features = ['use', 'temperature', 'humidity', 'windSpeed', 'pressure']  # Example weather-related columns
target = 'use'

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(smart_home_lstm[features])

# Create sequences for the last 24 hours to predict the next 1 hour
def create_sequences(smart_home_lstm, input_steps=24, output_steps=1):
    X, y = [], []
    for i in range(len(smart_home_lstm) - input_steps - output_steps + 1):
        X.append(smart_home_lstm[i:i+input_steps])
        y.append(smart_home_lstm[i+input_steps:i+input_steps+output_steps, 0])  # target is 'use'
    return np.array(X), np.array(y)

input_steps = 24
output_steps = 1
X, y = create_sequences(scaled_data, input_steps, output_steps)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to the original scale
y_test_scaled = scaler.inverse_transform(np.concatenate((y_test, np.zeros((y_test.shape[0], scaled_data.shape[1]-1))), axis=1))[:,0]
predictions_scaled = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1]-1))), axis=1))[:,0]


# Predict the next hour consumption based on the last 24 hours
last_24_hours = scaled_data[-24:].reshape(1, 24, len(features))
next_hour_prediction = model.predict(last_24_hours)

# Inverse transform the next hour prediction to the original scale
next_hour_prediction_scaled = scaler.inverse_transform(np.concatenate((next_hour_prediction, np.zeros((next_hour_prediction.shape[0], scaled_data.shape[1]-1))), axis=1))[:,0]

print(f'Predicted next hour consumption: {next_hour_prediction_scaled[0]}')


# In[62]:


# Plot the predictions vs actual values with datetime on the x-axis
plt.figure(figsize=(14, 5))
plt.plot(smart_home_lstm.index[-len(y_test_scaled):], y_test_scaled, color='blue', label='Actual Consumption')
plt.plot(smart_home_lstm.index[-len(y_test_scaled):], predictions_scaled, color='red', label='Predicted Consumption')
plt.title('Electricity Consumption Prediction')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.show()


# In[63]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# Calculate MAE
# Inverse transform the actual values
# Inverse transform the predictions and actual values to the original scale
y_test_scaled = scaler.inverse_transform(np.concatenate((y_test, np.zeros((y_test.shape[0], scaled_data.shape[1]-1))), axis=1))[:,0]


lstm_mae = mean_absolute_error(y_test_scaled, predictions_scaled)
lstm_mse = mean_squared_error(y_test_scaled, predictions_scaled)
lstm_rmse = np.sqrt(lstm_mse)
lstm_r2 = 1 - (np.sum((y_test_scaled - predictions_scaled)**2) / np.sum((y_test_scaled - np.mean(y_test_scaled))**2))
lstm_mape = mean_absolute_percentage_error(y_test_scaled, predictions_scaled)

print(f'Mean Absolute Error (MAE): {lstm_mae}')
print(f'Mean Squared Error (MSE): {lstm_mse}')
print(f'Root Mean Squared Error (RMSE): {lstm_rmse}')
print(f'R-squared (R2): {lstm_r2}')
print(f'MAPE: {lstm_mape:.2f}%')


# In[64]:


# Check if the predicted value exceeds the threshold
threshold = 4.0  # Threshold value in kW
if next_hour_prediction_scaled[0] > threshold:
    print("Predicted consumption exceeds the threshold. Optimizing values...")
    # Get the last row of the original data
    last_row = smart_home_lstm.iloc[-1]

    # Appliances with specific constraints
    appliances_06_09 = ['Furnace', 'Barn', 'Fridge']
    appliances_02_04 = ['Dishwasher', 'Home office', 'Wine cellar', 'Garage door', 'Well', 'Microwave', 'Living room', 'Kitchen']

    # Coefficients for the objective function
    c = np.ones(len(appliances_06_09) + len(appliances_02_04))

    # Bounds for each appliance
    bounds = [(0.6, 0.9) for _ in appliances_06_09] + [(0.2, 0.4) for _ in appliances_02_04]

    # Initial values from the last row
    x0 = np.array([last_row[appliance] for appliance in appliances_06_09 + appliances_02_04])

    # Linear programming to minimize the total consumption
    result = linprog(c, bounds=bounds, method='highs')

    # Extract the optimized values
    optimized_values = result.x

    # Print optimized values for each appliance
    print("Optimized appliance values:")
    for appliance, value in zip(appliances_06_09 + appliances_02_04, optimized_values):
        print(f"{appliance}: {value:.2f} kW")
        if value == 0:
            print(f"Please turn off {appliance} to achieve the optimal consumption.")

    # Calculate the new total consumption
    optimized_total = np.sum(optimized_values)
    print(f"Optimized total consumption: {optimized_total:.2f} kW")

    # Check if the optimized total is below the threshold
    if optimized_total > threshold:
        print("Even after optimization, the consumption is above the threshold. Consider reducing the use of major appliances or adjusting thermostat settings.")
    else:
        print("Optimization successful. The new consumption is within the threshold.")

    # Print the optimal value of use consumption
    print(f"Optimal value of use consumption: {optimized_total:.2f} kW")
else:
    print("Predicted consumption is within the threshold. No optimization needed.")

