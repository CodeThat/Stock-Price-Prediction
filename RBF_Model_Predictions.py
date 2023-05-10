import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg as backend
import sys


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
                self.dates.append(int(row[0].split('-')[0]))
                self.prices.append(float(row[1]))

    def predict_price(self, x):
        dates = np.array(self.dates).reshape(-1, 1)  # converting to matrix of n X 1
        prices = np.array(self.prices)

        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression model with RBF kernel
        svr_rbf.fit(dates, prices)  # fitting the data points to the RBF model

        return svr_rbf.predict([[x]])[0]


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

# Predict prices using RBF model
predicted_price = predictor.predict_price(prediction_date)

# Plot the predicted date
plt.scatter(prediction_date, predicted_price, color='red', label='Predicted Price')

# Add lines to the graph
plt.axvline(x=prediction_date, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=predicted_price, color='gray', linestyle='--', linewidth=1)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

print("Predicted Price (RBF model):", predicted_price)
