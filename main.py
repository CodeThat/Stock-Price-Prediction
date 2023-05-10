import csv
import numpy as np
from sklearn.svm import SVR

# Define stock price sheets and prediction date as variables
stock_price_sheet = 'Data\\NVDA.csv'
prediction_date = 7

dates = []
prices = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))


def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models
    svr_rbf.fit(dates, prices)  # fitting the data points in the models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


# Calling get_data method by passing the stock price sheet to it
get_data(stock_price_sheet)

# Predict prices
predicted_price = predict_price(dates, prices, prediction_date)
print("Predicted Prices:")
print("RBF model:", predicted_price[0])
print("Linear model:", predicted_price[1])
print("Polynomial model:", predicted_price[2])
