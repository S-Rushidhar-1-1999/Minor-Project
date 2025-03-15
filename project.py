import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fetch historical stock data for Wipro (symbol: WIPRO)
stock_data = pd.read_csv("S:/my_python/wipro-equity.csv", skipinitialspace=True)
stock_data.columns = stock_data.columns.str.strip().str.lower()

print(stock_data.head())

# Preprocessing: Feature Engineering
# Add moving averages to the dataset (adjusted windows)
stock_data['20_MA'] = stock_data['close'].rolling(window=20).mean()
stock_data['40_MA'] = stock_data['close'].rolling(window=40).mean()

# Check the number of rows before and after dropna
print(f"Number of rows before dropna: {len(stock_data)}")
stock_data = stock_data.dropna()
print(f"Number of rows after dropna: {len(stock_data)}")

# check if there is enough data.
if len(stock_data) < 2:
    print("Error: Not enough data after dropping NaN values. Please ensure your CSV file has sufficient data.")
    exit()

# Visualization of the closing price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(stock_data['close'], label='Wipro Closing Price', color='blue')
plt.plot(stock_data['20_MA'], label='20-Day Moving Average', color='red')
plt.plot(stock_data['40_MA'], label='40-Day Moving Average', color='green')
plt.legend()
plt.title('Wipro Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.show()

# Preparing data for prediction
# Using the closing price and moving averages to predict future prices
X = stock_data[['20_MA', '40_MA']]
y = stock_data['close']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

# Plot the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='red')
plt.legend()
plt.title('Wipro Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.show()

# Predicting the next day's price (example)
last_data = stock_data[['20_MA', '40_MA']].iloc[-1:].values
predicted_price = model.predict(last_data)
print(f"Predicted next day's price: {predicted_price[0]}")


"""
https://www.nseindia.com/get-quotes/equity?symbol=WIPRO 

EQ series 

Title --Stock Price Prediction

The art of forecasting stock prices has been a difficult task for many of the researchers and analysts. In fact, investors are highly interested in the research area of stock price prediction. For a good and successful investment, many investors are keen on knowing the future situation of the stock market. Good and effective prediction systems for the stock market help traders, investors, and analyst by providing supportive information like the future direction of the stock market.

need code for this project
"""
