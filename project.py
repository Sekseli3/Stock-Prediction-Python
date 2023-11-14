import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np  # Add this import
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM


#Fetching data from csv file 
data = pd.read_csv('MachineLearningINTC_data.csv')

#Date in data must be in form %d/%m%/%
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
data = data.sort_values(by='date')

 
#Select the training size to be 98% of the data, and the test size to be the rest. In addition, select the parts of the data we work with.  
train_size = int(0.98 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

#Resphape the matrices to compatible data. 
ytrain = ytrain.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)

#Choose the model and select variables. 
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
# Adds a dense layer with 25 neurons to the layer
model.add(Dense(25))
# Adds a dense layer with 1 neuron to the output layer
model.add(Dense(1))
#Views summary of training
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=32, epochs=100)

validation_loss = history.history['val_loss']

# Evaluate your model on the test set
test_loss = model.evaluate(xtest, ytest)
print("Test Loss:", test_loss)

# Make predictions
predictions = model.predict(xtest)

# Plot the actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(ytest, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Stock Prices')
plt.show()


test_data = test_data.copy()
test_data['predicted'] = predictions

# Plot the actual vs. predicted prices with date on the x-axis
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], test_data['close'], label='Actual', marker='o')
plt.plot(test_data['date'], test_data['predicted'], label='Predicted', marker='x')
plt.legend()
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
