import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load historical stock data for Wipro (symbol: WIPRO)
stock_data = pd.read_csv("wipro-equity.csv", skipinitialspace=True)
stock_data.columns = stock_data.columns.str.strip().str.lower()

# Feature Engineering: Add moving averages
stock_data['10_MA'] = stock_data['close'].rolling(window=10).mean()
stock_data['15_MA'] = stock_data['close'].rolling(window=15).mean()

# Check the number of rows before and after dropping NaN values
print(f"Number of rows before dropna: {len(stock_data)}")
stock_data = stock_data.dropna()
print(f"Number of rows after dropna: {len(stock_data)}")

# Ensure enough data remains after cleaning
if len(stock_data) < 2:
    print("Error: Not enough data after dropping NaN values. Please ensure your CSV file has sufficient data.")
    exit()

# Visualization of the closing price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(stock_data['close'], label='Wipro Closing Price', color='blue')
plt.plot(stock_data['10_MA'], label='10-Day Moving Average', color='red')
plt.plot(stock_data['15_MA'], label='15-Day Moving Average', color='green')
plt.legend()
plt.title('Wipro Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.show()

# Preparing data for prediction
X = stock_data[['10_MA', '15_MA']]
y = stock_data['close']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='red')
plt.legend()
plt.title('Wipro Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.show()

# Predicting the next day's price
last_data = pd.DataFrame(stock_data[['10_MA', '15_MA']].iloc[-1:])
predicted_price = model.predict(last_data)
print(f"Predicted next day's price: {predicted_price[0]}")
