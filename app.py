# -*- coding: utf-8 -*-

#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random, os
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
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-default-secret-key")

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
    symbol = request.form['symbol']
    
    if not symbol:
        flash('Please enter a stock symbol', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Get historical data
        get_historical(symbol)
        
        # Read the CSV
        df = pd.read_csv(f'{symbol}.csv')
        
        # Get tweet data
        tweets = retrieving_tweets_polarity(symbol)
        
        # Stock sentiment analysis
        stock_sentiment = retrieve_stock_sentiment(tweets)
        
        # Predict using different algorithms
        df_stock = pd.read_csv(f'{symbol}.csv')
        
        # ARIMA prediction
        arima_pred, error_arima = ARIMA_ALGO(df_stock)
        
        # LSTM prediction
        lstm_pred, error_lstm = LSTM_ALGO(df_stock)
        
        # Linear Regression prediction
        lr_pred, error_lr = LIN_REG_ALGO(df_stock)
        
        # Get the dataframe with prediction info
        df_pred = pd.DataFrame({
            'Model': ['ARIMA', 'LSTM', 'Linear Regression'],
            'Prediction': [arima_pred, lstm_pred, lr_pred],
            'RMSE': [error_arima, error_lstm, error_lr]
        })
        
        # Choose the best model based on RMSE
        best_model = df_pred.loc[df_pred['RMSE'].idxmin()]['Model']
        final_pred = df_pred.loc[df_pred['RMSE'].idxmin()]['Prediction']
        
        return render_template('prediction.html', 
                             symbol=symbol.upper(),
                             prediction=round(final_pred, 2),
                             df_pred=df_pred,
                             best_model=best_model,
                             sentiment=stock_sentiment)
    
    except Exception as e:
        flash(f'Error processing request: {str(e)}', 'danger')
        logging.error(f"Error in prediction: {str(e)}")
        return redirect(url_for('index'))

#**************** FUNCTIONS TO FETCH DATA ***************************
def get_historical(quote):
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)
    
    try:
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(f'{quote}.csv')
        
        if df.empty:
            ts = TimeSeries(key=os.environ.get('ALPHA_VANTAGE_KEY', 'N6A6QT6IBFJOPJ70'), output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
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
        logging.error(f"Error fetching historical data: {str(e)}")
        raise

#******************** ARIMA SECTION ********************
def ARIMA_ALGO(df):
    try:
        # Prepare data
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # For daily basis
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')
        
        df['Price'] = df['Close']
        Quantity_date = df[['Price']]
        
        # Fill missing values
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        
        # Plot trends
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(Quantity_date)
        plt.savefig('static/Trends.png')
        plt.close(fig)
        
        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]
        
        # ARIMA model function
        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions
        
        # Fit in model
        predictions = arima_model(train, test)
        
        # Plot graph
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/ARIMA.png')
        plt.close(fig)
        
        # Get prediction
        arima_pred = predictions[-1]
        
        # RMSE calculation
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        
        return arima_pred, error_arima
    except Exception as e:
        logging.error(f"Error in ARIMA prediction: {str(e)}")
        raise

#************* LSTM SECTION **********************
def LSTM_ALGO(df):
    try:
        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Dropout
        from sklearn.preprocessing import MinMaxScaler
        
        # Split data into training set and test set
        dataset_train = df.iloc[0:int(0.8*len(df)),:]
        dataset_test = df.iloc[int(0.8*len(df)):,:]
        
        training_set = df.iloc[:,4:5].values  # Close price column
        
        # Feature Scaling
        sc = MinMaxScaler(feature_range=(0,1))
        training_set_scaled = sc.fit_transform(training_set)
        
        # Creating data structure with 7 timesteps and 1 output
        X_train = []
        y_train = []
        for i in range(7, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        X_forecast = np.array(X_train[-1,1:])
        X_forecast = np.append(X_forecast, y_train[-1])
        
        # Reshaping: Adding 3rd dimension
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
        
        # Building RNN
        regressor = Sequential()
        
        # Add LSTM layers and Dropout regularization
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        regressor.add(Dropout(0.1))
        
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
        # Add output layer
        regressor.add(Dense(units=1))
        
        # Compile and fit model
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressor.fit(X_train, y_train, epochs=25, batch_size=32)
        
        # Testing
        real_stock_price = dataset_test.iloc[:,4:5].values
        
        # To predict, we need stock prices of 7 days before the test set
        dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
        testing_set = dataset_total[len(dataset_total)-len(dataset_test)-7:].values
        testing_set = testing_set.reshape(-1,1)
        
        # Feature scaling
        testing_set = sc.transform(testing_set)
        
        # Create data structure
        X_test = []
        for i in range(7, len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
            
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Testing Prediction
        predicted_stock_price = regressor.predict(X_test)
        
        # Getting original prices back from scaled values
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        # Plot results
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        # Calculate RMSE
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        
        # Forecasting Prediction
        forecasted_stock_price = regressor.predict(X_forecast)
        
        # Getting original prices back from scaled values
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
        
        lstm_pred = forecasted_stock_price[0,0]
        
        return lstm_pred, error_lstm
    except Exception as e:
        logging.error(f"Error in LSTM prediction: {str(e)}")
        raise

#***************** LINEAR REGRESSION SECTION ******************
def LIN_REG_ALGO(df):
    try:
        # No of days to be forecasted in future
        forecast_out = 7
        
        # Prepare data
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date (newest to oldest)
        df = df.sort_values('Date')
        
        # Create target variable
        df['Prediction'] = df['Close'].shift(-forecast_out)
        
        # Create the independent data set (X)
        # Convert the dataframe to a numpy array
        X = np.array(df.drop(['Prediction'], 1))
        
        # Remove the last n rows
        X = X[:-forecast_out]
        
        # Create the dependent data set (y)
        # Convert the dataframe to a numpy array
        y = np.array(df['Prediction'])
        
        # Remove the last n rows
        y = y[:-forecast_out]
        
        # Split the data into 80% training and 20% testing
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Create and train the Linear Regression Model
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        
        # Testing Model: Score returns the coefficient of determination R^2
        lr_confidence = lr.score(x_test, y_test)
        
        # Set X_forecast equal to the last n rows of the original data set
        X_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        
        # Print linear regression prediction
        lr_prediction = lr.predict(X_forecast)
        
        # Calculate RMSE
        y_test_pred = lr.predict(x_test)
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Plot results
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(df['Close'].tail(30), label='Actual Price')
        plt.plot(pd.DataFrame({'Prediction': lr_prediction}, 
                              index=df['Date'].tail(forecast_out)), label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/LINEAR_REG.png')
        plt.close(fig)
        
        # Get the prediction for tomorrow
        lr_pred = lr_prediction[0]
        
        return lr_pred, error_lr
    except Exception as e:
        logging.error(f"Error in Linear Regression prediction: {str(e)}")
        raise

#***************** TWITTER SENTIMENT ANALYSIS ******************
def retrieving_tweets_polarity(symbol):
    try:
        # Twitter API Authentication
        auth = tweepy.OAuthHandler(
            os.environ.get('TWITTER_API_KEY', ct.consumer_key),
            os.environ.get('TWITTER_API_SECRET', ct.consumer_secret)
        )
        auth.set_access_token(
            os.environ.get('TWITTER_ACCESS_TOKEN', ct.access_token),
            os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', ct.access_token_secret)
        )
        user = tweepy.API(auth)
        
        # Search term
        search_term = f"#{symbol} OR ${symbol} -filter:retweets"
        
        # Get tweets
        tweets = tweepy.Cursor(user.search_tweets, q=search_term, lang='en', tweet_mode='extended').items(100)
        
        # Create a list of polarity values for each tweet
        tweet_list = []
        global_polarity = 0
        
        for tweet in tweets:
            tw = {}
            tw['text'] = tweet.full_text
            tw['processed_text'] = p.clean(tweet.full_text)
            
            # Clean the tweet for sentiment analysis
            tw['processed_text'] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tw['processed_text']).split())
            
            # Get sentiment analysis scores
            analysis = TextBlob(tw['processed_text'])
            tw['polarity'] = analysis.sentiment.polarity
            tw['subjectivity'] = analysis.sentiment.subjectivity
            
            # Add to global polarity
            global_polarity += tw['polarity']
            
            tweet_list.append(tw)
        
        # Calculate average polarity
        if len(tweet_list) > 0:
            global_polarity = global_polarity / len(tweet_list)
        
        return tweet_list
    except Exception as e:
        logging.error(f"Error retrieving tweets: {str(e)}")
        return []

def retrieve_stock_sentiment(tweets):
    try:
        # Calculate average polarity
        if not tweets or len(tweets) == 0:
            return "Neutral"
            
        total_polarity = sum(tweet['polarity'] for tweet in tweets)
        avg_polarity = total_polarity / len(tweets)
        
        # Determine sentiment
        if avg_polarity >= 0.05:
            return "Positive"
        elif avg_polarity <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logging.error(f"Error calculating stock sentiment: {str(e)}")
        return "Neutral"
