# Tesla Stock Price Prediction using LSTM

This project demonstrates how to use Long Short-Term Memory (LSTM) networks to predict Tesla stock prices based on historical stock data. LSTMs are powerful for time-series forecasting, making them highly effective for stock price prediction.

## Project Overview

This project focuses on predicting Tesla's stock prices using historical stock data and a Long Short-Term Memory (LSTM) neural network. LSTMs are highly effective for time-series forecasting tasks as they can capture temporal dependencies and patterns in sequential data, making them ideal for stock price prediction.

The main goal of this project is to predict Tesla’s stock closing prices based on past prices, allowing for better insight into future price movements. Accurate stock price prediction can be beneficial for making informed investment decisions, analyzing market trends, and managing risk.

We chose LSTM because it can retain information over long sequences, making it suitable for modeling stock market data, which is known for its volatility and time-dependent characteristics.

### Steps:
1. **Data Collection**: Using `yfinance` to collect Tesla stock price data.
2. **Data Preprocessing**: Normalizing the data and creating a look-back window to predict future stock prices.
3. **Model Building**: Using Keras to build a multi-layer LSTM model.
4. **Training and Evaluation**: Training the model and evaluating its performance using metrics such as MSE, RMSE, and MAE.
5. **Visualization**: Comparing the predicted stock prices with the actual stock prices to evaluate model performance.

## Dataset

The dataset was sourced from [Yahoo Finance](https://finance.yahoo.com/), containing historical data for Tesla stocks.

## Model Architecture

The model consists of:
- Two LSTM layers with 128 and 64 units, followed by Dropout layers to prevent overfitting.
- The final output layer is a dense layer that predicts the closing price.

## Results

The model's performance was evaluated using the following metrics:
- **Mean Squared Error (MSE)**: X.XX
- **Root Mean Squared Error (RMSE)**: X.XX
- **Mean Absolute Error (MAE)**: X.XX

### Example Plot: Actual vs Predicted Stock Prices

![Stock Price Prediction Plot](results/stock_prediction.png)

## Setup

To run this project locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/himanshu-dandle/tesla-stock-lstm.git
