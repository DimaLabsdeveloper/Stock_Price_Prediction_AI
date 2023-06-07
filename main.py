import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Define the ticker symbol for BTC
ticker_symbol = "BTC-USD"

# Download the historical data for BTC for the last 7 days with 1-hour intervals
end_time = datetime.now()
start_time = end_time - timedelta(days=7)
stock_data = yf.download(ticker_symbol, start=start_time, end=end_time, interval="1h")

# Create a DataFrame with the 'Close' prices as the target variable
data = pd.DataFrame(stock_data["Close"])

# Create a new column with the target variable shifted one period ahead
data["Target"] = data["Close"].shift(-1)

# Drop the last row which contains NaN values
data = data[:-1]

# Create a new DataFrame for predictions
predictions_df = pd.DataFrame(index=data.index, columns=["Actual", "Predicted"])

# Iterate over each row in the data DataFrame
for i, row in data.iterrows():
    # Create a subset of the data for training
    train_data = data.loc[:i]

    # Split the data into features (X) and target (y)
    X_train = train_data[["Close"]]
    y_train = train_data["Target"]

    # Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make a prediction for the next hour
    next_hour = pd.DataFrame([[row["Close"]]], columns=["Close"])
    prediction = model.predict(next_hour)

    # Update the predictions DataFrame
    predictions_df.loc[i, "Actual"] = row["Target"]
    predictions_df.loc[i, "Predicted"] = prediction[0]

# Make future predictions
def createfile():
    future_predictions_df = pd.DataFrame(columns=["Date", "Close", "Target"])
    end_time_future = end_time + timedelta(hours=1)
    for j in range(24):  # Predict next 24 hours
        future_data = pd.concat([data, future_predictions_df])
        X_future = future_data[["Close"]]
        model.fit(X_future, future_data["Target"])
        next_hour_future = pd.DataFrame([[future_data.iloc[-1]["Close"]]], columns=["Close"])
        future_prediction = model.predict(next_hour_future)
        future_predictions_df.loc[end_time_future, "Date"] = end_time_future
        future_predictions_df.loc[end_time_future, "Close"] = future_data.iloc[-1]["Close"]
        future_predictions_df.loc[end_time_future, "Target"] = future_prediction[0]
        end_time_future += timedelta(hours=1)

# Save future predictions to a text file
    future_predictions_df.to_csv("future_predictions.txt", sep="\t", index=False)

choise = int(input("1 - create file with predictions\n2 - give opinion about price\n3 - close program\nWrite your number:"))
if choise == 1:
    createfile()
elif choise==2:
    print("soon")
elif choise == 3:
    sys.exit()

