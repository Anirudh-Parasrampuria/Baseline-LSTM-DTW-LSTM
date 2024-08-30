import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import random

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(2055)

# Load the dataset
data = pd.read_csv('/Users/anirudhparasramouria/Downloads/ETH-USD.csv')  # Replace with your dataset path
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Add other features, e.g., moving averages
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_10'] = data['Close'].rolling(window=10).mean()

# Drop NaN values
data = data.dropna()

# Use relevant features
features = ['Close', 'MA_5', 'MA_10']
data = data[features]

# Convert to numpy array
dataset = data.values

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Function to create a dataset with look_back time steps
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])  # Predict the close price
    return np.array(X), np.array(Y)

look_back = 60  # Number of previous time steps to use as input variables
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, len(features)))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, len(features)))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, len(features))))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, batch_size=1, epochs=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and actual values
train_predict = scaler.inverse_transform(np.hstack((train_predict, np.zeros((train_predict.shape[0], len(features)-1)))))
test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], len(features)-1)))))
Y_train = scaler.inverse_transform(np.hstack((Y_train.reshape(-1, 1), np.zeros((Y_train.shape[0], len(features)-1)))))
Y_test = scaler.inverse_transform(np.hstack((Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], len(features)-1)))))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(Y_train[:, 0], train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(Y_test[:, 0], test_predict[:, 0]))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Plotting
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[look_back:len(train_predict) + look_back, 0] = train_predict[:, 0]

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (look_back * 2) + 1:len(scaled_data) - 1, 0] = test_predict[:, 0]

plt.figure(figsize=(16, 8))
plt.plot(data.index, scaler.inverse_transform(scaled_data)[:, 0], label='Original data')
plt.plot(data.index, train_plot[:, 0], label='Training prediction')
plt.plot(data.index, test_plot[:, 0], label='Testing prediction')
plt.legend()
plt.show()
