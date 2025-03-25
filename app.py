# -*- coding: utf-8 -*-

#**************** IMPORT PACKAGES ********************
import os
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
import nltk
nltk.download('punkt', quiet=True)
import logging

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#***************** FLASK *****************************
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.form['stock_symbol']
        
        # Check if the input is not empty
        if not stock_symbol:
            flash('Please enter a stock symbol', 'danger')
            return redirect(url_for('index'))
        
        # Get stock data
        get_historical(stock_symbol)
        
        # Read the data
        df = pd.read_csv(f'{stock_symbol}.csv')
        
        # Calculate predictions using different models
        arima_pred, arima_rmse = ARIMA_ALGO(df)
        lstm_pred, lstm_rmse = LSTM_ALGO(df)
        lin_reg_pred, lin_reg_rmse, forecast_set = LIN_REG_ALGO(df)
        
        # Perform sentiment analysis on tweets
        tweet_sentiment, tweet_polarity = get_tweet_sentiment(stock_symbol)
        
        # Get last closing price
        last_price = df['Close'].iloc[-1]
        
        # Calculate average prediction with weighted average
        weight_arima = 1.0 / arima_rmse if arima_rmse > 0 else 0
        weight_lstm = 1.0 / lstm_rmse if lstm_rmse > 0 else 0
        weight_lin_reg = 1.0 / lin_reg_rmse if lin_reg_rmse > 0 else 0
        weight_sentiment = 0.2  # Fixed weight for sentiment
        
        # Adjust prediction based on sentiment
        sentiment_adjustment = 0.01 * tweet_polarity * last_price
        
        # Calculate weighted average prediction
        total_weight = weight_arima + weight_lstm + weight_lin_reg + weight_sentiment
        weighted_pred = (weight_arima * arima_pred + 
                         weight_lstm * lstm_pred + 
                         weight_lin_reg * lin_reg_pred + 
                         weight_sentiment * (last_price + sentiment_adjustment)) / total_weight
        
        # Calculate % change
        change_arima = ((arima_pred - last_price) / last_price) * 100
        change_lstm = ((lstm_pred - last_price) / last_price) * 100
        change_lin_reg = ((lin_reg_pred - last_price) / last_price) * 100
        change_weighted = ((weighted_pred - last_price) / last_price) * 100
        
        return render_template('prediction.html', 
                              stock_symbol=stock_symbol,
                              last_price=round(last_price, 2),
                              arima_pred=round(arima_pred, 2),
                              lstm_pred=round(lstm_pred, 2),
                              lin_reg_pred=round(lin_reg_pred, 2),
                              weighted_pred=round(weighted_pred, 2),
                              change_arima=round(change_arima, 2),
                              change_lstm=round(change_lstm, 2),
                              change_lin_reg=round(change_lin_reg, 2),
                              change_weighted=round(change_weighted, 2),
                              arima_rmse=round(arima_rmse, 2),
                              lstm_rmse=round(lstm_rmse, 2),
                              lin_reg_rmse=round(lin_reg_rmse, 2),
                              tweet_sentiment=tweet_sentiment,
                              tweet_polarity=round(tweet_polarity, 2),
                              forecast_set=forecast_set.tolist() if forecast_set is not None else [])
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        flash(f'Error in prediction: {str(e)}', 'danger')
        return redirect(url_for('index'))

#**************** FUNCTIONS TO FETCH DATA ***************************
def get_historical(quote):
    try:
        end = datetime.now()
        start = datetime(end.year-2, end.month, end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(f'{quote}.csv')
        
        if df.empty:
            # Fallback to Alpha Vantage if Yahoo Finance fails
            ts = TimeSeries(key=os.environ.get('ALPHA_VANTAGE_KEY', 'N6A6QT6IBFJOPJ70'), output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol=f'NSE:{quote}', outputsize='full')
            
            # Format df - Last 2 yrs rows => 502, in ascending order => ::-1
            data = data.head(503).iloc[::-1]
            data = data.reset_index()
            
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Adj Close'] = data['5. adjusted close']
            df['Volume'] = data['6. volume']
            df.to_csv(f'{quote}.csv', index=False)
            
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise Exception(f"Error fetching historical data: {str(e)}")

#******************** ARIMA SECTION ********************
def ARIMA_ALGO(df):
    try:
        # Ensure Date column is present and properly formatted
        if 'Date' not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Prepare data for ARIMA
        data = df[['Date', 'Close']].copy()
        data.set_index('Date', inplace=True)
        
        # Split into train and test
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]
        
        # Plot the historical data
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(data)
        plt.title(f'Stock Price History')
        plt.savefig('static/Trends.png')
        plt.close(fig)
        
        # ARIMA forecasting
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Forecasting
        forecast = model_fit.forecast(steps=len(test))
        
        # Plot
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, label='Actual Price')
        plt.plot(test.index, forecast, label='Predicted Price')
        plt.legend(loc=4)
        plt.title('ARIMA Model Prediction')
        plt.savefig('static/ARIMA.png')
        plt.close(fig)
        
        # Calculate RMSE
        rmse = math.sqrt(mean_squared_error(test, forecast))
        
        # Predict next day's price
        model = ARIMA(data, order=(5, 1, 0))
        model_fit = model.fit()
        next_day_forecast = model_fit.forecast(steps=1)
        arima_pred = next_day_forecast[0]
        
        logger.info(f"ARIMA Prediction: {arima_pred}, RMSE: {rmse}")
        return arima_pred, rmse
    
    except Exception as e:
        logger.error(f"Error in ARIMA model: {str(e)}")
        # Return default values if ARIMA fails
        return df['Close'].iloc[-1], 999999

#************* LSTM SECTION **********************
def LSTM_ALGO(df):
    try:
        # Import required libraries for LSTM
        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Dropout
        from sklearn.preprocessing import MinMaxScaler
        
        # Prepare data
        df_lstm = df[['Close']].copy()
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_lstm)
        
        # Split into train and test sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Function to create dataset with time steps
        def create_dataset(data, time_steps=1):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:(i + time_steps), 0])
                y.append(data[i + time_steps, 0])
            return np.array(X), np.array(y)
        
        # Time steps (look back period)
        time_steps = 60
        
        # Create the training dataset
        X_train, y_train = create_dataset(train_data, time_steps)
        X_test, y_test = create_dataset(test_data, time_steps)
        
        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
        
        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Invert predictions to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate RMSE
        rmse = math.sqrt(mean_squared_error(y_test_scaled, test_predict))
        
        # Plot LSTM predictions
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(df_lstm.iloc[train_size+time_steps:train_size+time_steps+len(test_predict)].index, 
                 y_test_scaled, label='Actual Price')
        plt.plot(df_lstm.iloc[train_size+time_steps:train_size+time_steps+len(test_predict)].index, 
                 test_predict, label='Predicted Price')
        plt.legend(loc=4)
        plt.title('LSTM Model Prediction')
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        # Predict next day's price
        # Get the last time_steps days of data
        last_data = scaled_data[-time_steps:]
        last_data = np.reshape(last_data, (1, time_steps, 1))
        
        # Predict
        next_day_pred = model.predict(last_data)
        next_day_pred = scaler.inverse_transform(next_day_pred)[0, 0]
        
        logger.info(f"LSTM Prediction: {next_day_pred}, RMSE: {rmse}")
        return next_day_pred, rmse
    
    except Exception as e:
        logger.error(f"Error in LSTM model: {str(e)}")
        # Return default values if LSTM fails
        return df['Close'].iloc[-1], 999999

#***************** LINEAR REGRESSION SECTION ******************       
def LIN_REG_ALGO(df):
    try:
        # Prepare data
        # Create features
        df = df.copy()
        
        # Add indicators
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
        df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
        
        # Keep only relevant columns
        df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
        
        # Forecast period
        forecast_out = 7  # 7 days
        df['label'] = df['Close'].shift(-forecast_out)
        
        # Split data
        X = np.array(df.drop(['label'], axis=1))
        X = X[:-forecast_out]
        y = np.array(df['label'].dropna())
        
        # Train/test split
        X_train, X_test, y_train, y_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):], y[:int(0.8*len(y))], y[int(0.8*len(y)):]
        
        # Train Linear Regression model
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Test accuracy
        accuracy = clf.score(X_test, y_test)
        
        # Predict next days
        X_forecast = np.array(df.drop(['label'], axis=1))[-forecast_out:]
        forecast_set = clf.predict(X_forecast)
        
        # Calculate RMSE
        y_pred = clf.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        
        # Plot
        df['Forecast'] = np.nan
        last_date = df.index[-forecast_out]
        
        # If index is not datetime, convert it
        if not isinstance(last_date, datetime) and not isinstance(last_date, pd.Timestamp):
            last_date = df.index[-1]
            
        df.iloc[-forecast_out:, df.columns.get_loc('Forecast')] = forecast_set
        
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(df['Close'], label='Historical Data')
        plt.plot(df['Forecast'], label='Forecast')
        plt.legend(loc=4)
        plt.title('Linear Regression Forecast')
        plt.savefig('static/Linear_Regression.png')
        plt.close(fig)
        
        logger.info(f"Linear Regression Prediction (next day): {forecast_set[0]}, RMSE: {rmse}")
        return forecast_set[0], rmse, forecast_set
    
    except Exception as e:
        logger.error(f"Error in Linear Regression model: {str(e)}")
        # Return default values if Lin Reg fails
        return df['Close'].iloc[-1], 999999, np.array([df['Close'].iloc[-1]] * 7)

#*************** TWITTER SENTIMENT ANALYSIS *******************
def get_tweet_sentiment(quote):
    try:
        # Setup tweepy
        consumer_key = os.environ.get('TWITTER_CONSUMER_KEY', ct.consumer_key)
        consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET', ct.consumer_secret)
        access_token = os.environ.get('TWITTER_ACCESS_TOKEN', ct.access_token)
        access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', ct.access_token_secret)
        
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Search term
        search_term = f"{quote} stock OR ${quote}"
        
        # Get tweets
        tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang="en", result_type="recent", count=100).items(100)
        
        # Process tweets
        tweet_list = []
        for tweet in tweets:
            # Clean tweet text
            clean_text = p.clean(tweet.text)
            # Remove URLs, mentions, hashtags
            clean_text = re.sub(r'http\S+|www\S+|https\S+', '', clean_text, flags=re.MULTILINE)
            clean_text = re.sub(r'@\w+|\#', '', clean_text)
            
            # Get sentiment
            analysis = TextBlob(clean_text)
            sentiment = analysis.sentiment.polarity
            
            tweet_obj = Tweet(tweet.id, clean_text, sentiment)
            tweet_list.append(tweet_obj)
        
        # Calculate average sentiment
        if tweet_list:
            avg_polarity = sum(tweet.polarity for tweet in tweet_list) / len(tweet_list)
        else:
            avg_polarity = 0
            
        # Determine sentiment category
        if avg_polarity > 0.1:
            sentiment_category = "Positive"
        elif avg_polarity < -0.1:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"
            
        logger.info(f"Twitter Sentiment for {quote}: {sentiment_category} (Polarity: {avg_polarity})")
        return sentiment_category, avg_polarity
        
    except Exception as e:
        logger.error(f"Error in Twitter sentiment analysis: {str(e)}")
        return "Neutral", 0
