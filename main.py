import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, timedelta
from textblob import TextBlob

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

# Perform sentiment analysis on tweets
def perform_sentiment_analysis(tweets):
    sentiment_scores = []
    for tweet in tweets:
        blob = TextBlob(tweet)
        sentiment_scores.append(blob.sentiment.polarity)
    return sentiment_scores

# Fetch Twitter data and analyze opinions
def fetch_twitter_data():
    # Fetch tweets (replace with your own logic to fetch tweets)
    tweets = [
        "BTC is doing great! #Bitcoin",
        "I think BTC will go up in the next hour.",
        "Avoid BTC, it's going down!",
        "BTC price is stable.",
        "Just bought some BTC, expecting a rise!"
    ]

    sentiment_scores = perform_sentiment_analysis(tweets)
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    return average_sentiment

# Make future predictions
def create_file_with_predictions():
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

def check_buy_signal(predictions_df):
    """
    Check if it is a good moment to buy BTC based on the predictions.

    Arguments:
    predictions_df -- DataFrame with actual and predicted BTC prices.

    Returns:
    buy_signal -- Boolean value indicating whether it is a good moment to buy BTC.
    """
    last_actual_price = predictions_df.iloc[-1]["Actual"]
    last_predicted_price = predictions_df.iloc[-1]["Predicted"]

    # Define a threshold for considering it a good moment to buy BTC
    threshold = 0.0  # Modify this threshold according to your preference

    if last_predicted_price > last_actual_price + threshold:
        buy_signal = True
    else:
        buy_signal = False

    return buy_signal


# Main program
while True:
    choice = int(input("1 - Create file with predictions\n2 - Give opinion about price\n3 - Close program\nWrite your number: "))

    if choice == 1:
        create_file_with_predictions()
        print("File with future predictions created.\n")

    elif choice == 2:
        sentiment = fetch_twitter_data()
        buy_signal = check_buy_signal(predictions_df)
        if buy_signal and sentiment > 0:
            print("It is a good moment to buy BTC.\n")
        else:
            print("It is not a good moment to buy BTC.\n")

    elif choice == 3:
        break

    else:
        print("Invalid choice. Please select a valid option.\n")

print("Program closed.")
