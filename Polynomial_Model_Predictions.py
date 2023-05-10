import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg as backend
import matplotlib.dates as mdates
from datetime import datetime


class StockPricePredictor:
    def __init__(self, filename):
        self.dates = []
        self.prices = []
        self.load_data(filename)

    def load_data(self, filename):
        with open(filename, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)  # skipping column names
            for row in csvFileReader:
                date = datetime.strptime(row[0], '%Y-%m-%d')
                self.dates.append(date)
                self.prices.append(float(row[1]))

    def predict_price(self, x):
        dates = np.array([date.timestamp() for date in self.dates]).reshape(-1, 1)  # converting to matrix of n X 1
        prices = np.array(self.prices)

        svr_lin = SVR(kernel='linear', C=1e3)  # defining the support vector regression model with linear kernel
        svr_lin.fit(dates, prices)  # fitting the data points to the linear model

        return svr_lin.predict([[x]])[0]


# Define stock price sheets and prediction date as variables
stock_price_sheet = 'Data\\NVDA.csv'
prediction_date = 7

# Create the predictor
predictor = StockPricePredictor(stock_price_sheet)

# Use Qt5 backend for faster and interactive plots
fig = plt.figure()
backend.FigureCanvasQTAgg(fig)

# Plot the chart
dates = predictor.dates
prices = predictor.prices

plt.scatter(dates, prices, color='black', label='Data')  # plotting the initial datapoints

# Predict prices using linear model
predicted_price = predictor.predict_price(prediction_date)

# Plot the predicted date
prediction_date = dates[prediction_date]
plt.scatter(prediction_date, predicted_price, color='red', label='Predicted Price')

# Add lines to the graph
plt.axvline(x=prediction_date, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=predicted_price, color='gray', linestyle='--', linewidth=1)

# Set the date formatter on the X-axis
date_format = mdates.DateFormatter('%m/%d/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.gcf().autofmt_xdate()  # Rotate and align the dates on the X-axis for better visibility
plt.show()

print("Predicted Price (Linear model):", predicted_price)
