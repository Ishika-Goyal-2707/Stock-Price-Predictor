# stock_price_predictor.py
# ------------------------------------------
# Simple Stock Price Prediction using Linear Regression
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Download Stock Data
ticker = input("Enter Stock Symbol (e.g. AAPL, TSLA, RELIANCE.NS): ")
data = yf.download(ticker, start='2018-01-01', end='2025-01-01')

if data.empty:
    print("❌ No data found! Check the stock symbol and try again.")
    exit()

print("\n✅ Data successfully fetched!")
print(data.head())

# Step 2: Prepare Data
data = data[['Close']]
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = np.array(data[['Close']])
y = np.array(data['Target'])

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n📊 Model Evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Step 7: Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 8: Predict Next Day Price
latest_price = data['Close'].iloc[-1]
next_day_price = model.predict([[latest_price]])[0]
print(f"\n💰 Predicted Next Day Closing Price for {ticker}: ${next_day_price:.2f}")
