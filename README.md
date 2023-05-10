# Stock Price Prediction

This project uses support vector regression (SVR) models to predict stock prices based on historical data. It employs different kernels, such as linear, polynomial, and radial basis function (RBF), to train the models and make predictions.

## Installation

1. Clone the repository:
```git clone https://github.com/your-username/stock-price-prediction.git```

2. Install the required dependencies:

```pip install numpy sklearn matplotlib```

## Usage

1. Prepare your stock price data in CSV format and place it in the `Data` directory.

2. Modify the `stock_price_sheet` variable in the script (`main.py`) to specify the path to your CSV file.

3. Optionally, adjust the `prediction_date` variable to set the prediction date.

4. Run the script:
```python main.py```

5. The predicted prices for the RBF, linear, and polynomial models will be printed to the terminal.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).



