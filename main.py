import warnings

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import yfinance as yf
from datetime import datetime, timedelta
from pytz import timezone

from textblob import TextBlob

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Define the timezone
timezone_name = timezone('America/New_York')  # Replace "your_timezone" with the appropriate timezone name

# Get the timezone object
tz = timezone('Europe/Madrid')

# Define the ticker symbol for BTC
ticker_symbol = "BTC-USD"

# Convert the end_time to the specified timezone
end_time = datetime.now(tz)

# Download the historical data for BTC for the last 7 days with 1-hour intervals
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
        "Just bought some BTC, expecting a rise!",
        "BTC is experiencing a bullish trend.",
        "Expecting a significant surge in BTC price soon.",
        "Positive market sentiment towards BTC.",
        "BTC showing signs of strong upward momentum.",
        "Forecasts indicate a potential price increase for BTC.",
        "Cautiously optimistic about BTC's future performance.",
        "Anticipating a positive shift in BTC's value.",
        "Market experts predict a bullish run for BTC.",
        "Observing a steady rise in BTC's market dominance.",
        "Investors are increasingly interested in BTC.",
        "BTC is gaining momentum, poised for growth!",
        "Positive indicators point towards a promising future for BTC.",
        "Investment opportunities for traders.",
        "Institutional interest in BTC continues to rise.",
        "Analyzing market trends for BTC price predictions.",
        "#Bitcoin"
    ]

    sentiment_scores = perform_sentiment_analysis(tweets)
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    return average_sentiment


# Check if it is a good moment to buy BTC based on the predictions and sentiment analysis
def check_buy_signal(predictions_df, sentiment):
    last_actual_price = predictions_df.iloc[-1]["Actual"]
    last_predicted_price = predictions_df.iloc[-1]["Predicted"]

    threshold = 0.0  # Modify this threshold according to your preference

    if last_predicted_price > last_actual_price + threshold and sentiment > 0:
        buy_signal = True
    else:
        buy_signal = False

    return buy_signal


# Create file with future predictions
def create_file_with_predictions():
    future_predictions_df = pd.DataFrame(columns=["Date", "Close", "Target"])
    end_time_future = end_time + timedelta(hours=1)
    future_data = data.copy()  # Copy the historical data to the future_data DataFrame

    # Create a SimpleImputer to handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_future = future_data[["Close"]]

    # Fit the imputer on the historical data
    imputer.fit(X_future)

    for j in range(24):  # Predict next 24 hours
        next_hour_future = pd.DataFrame([[future_data.iloc[-1]["Close"]]], columns=["Close"])

        # Impute missing values in next_hour_future
        next_hour_future = imputer.transform(next_hour_future)

        future_prediction = model.predict(next_hour_future)
        future_predictions_df.loc[end_time_future, "Date"] = end_time_future
        future_predictions_df.loc[end_time_future, "Close"] = future_data.iloc[-1]["Close"]
        future_predictions_df.loc[end_time_future, "Target"] = future_prediction[0]

        # Update future_data DataFrame with the new prediction
        future_data = future_data.loc[future_data.index[-1]]  # Extract the last row as a Series
        future_data["Close"] = future_data["Target"]  # Update the Close value
        future_data["Target"] = future_prediction[0]  # Update the Target value
        future_data = future_data.to_frame().T  # Convert the Series back to a DataFrame
        future_data.index = [end_time_future]  # Update the index to the new timestamp

        end_time_future += timedelta(hours=1)

    # Save future predictions to a text file
    future_predictions_df.to_csv("future_predictions.txt", sep="\t", index=False)


# Plot the predicted prices
def plot_predicted_prices(predictions_df):
    plt.plot(predictions_df.index, predictions_df["Predicted"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("BTC Predicted Prices")
    plt.legend()
    plt.show()


# Main program
while True:
    choice = int(input("1 - Create file with predictions\n2 - Give opinion about price\n3 - Make a graph of predicted prices\n4 - Close program\nWrite your number: "))

    if choice == 1:
        create_file_with_predictions()
        print("File with future predictions created.\n")

    elif choice == 2:
        sentiment = fetch_twitter_data()
        buy_signal = check_buy_signal(predictions_df, sentiment)
        if buy_signal:
            print("It is a good moment to buy BTC.\n")
        else:
            print("It is not a good moment to buy BTC.\n")

    elif choice == 3:
        plot_predicted_prices(predictions_df)

    elif choice == 4:
        break

    else:
        print("Invalid choice. Please try again.\n")
