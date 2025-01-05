import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


# Function to create time-delay embedding (lagged features)
def time_delay_embedding(data, lag):
    """
    return: Embedded matrix of shape (len(data) - lag, lag).
    """
    return np.array([data[i: i + lag] for i in range(len(data) - lag)])

# Kernel Analog Forecasting
def kernel_analog_forecasting(data, lag, forecast_steps, bandwidth=1.0):
    """
    Perform Kernel Analog Forecasting (KAF) on a univariate time series.
    :param data: Array of time series data.
    :param lag: Number of lags for embedding.
    :param forecast_steps: Number of steps to forecast.
    :param bandwidth: Kernel bandwidth parameter.
    :return: Forecasted values for the specified number of steps.
    """
    lag_matrix = time_delay_embedding(data, lag)
    train = lag_matrix[:-1]  
    target = lag_matrix[1:, -1]  

    predictions = []
    for i in range(len(data) - lag - forecast_steps):
        query_point = lag_matrix[i].reshape(1, -1)  # Current state
        weights = rbf_kernel(query_point, train, gamma=1 / (2 * bandwidth**2))
        weights = weights.flatten() / weights.sum()  # Normalize weights

        # Forecast the next value as a weighted sum of historical analogs
        next_value = np.dot(weights, target)
        predictions.append(next_value)

    return predictions

# Backtesting function
def kaf(data, lag, forecast_steps, bandwidth):
    """
    Backtest the KAF model by comparing predictions with actual values.
    :param data: Time series data.
    :param lag: Number of lags for embedding.
    :param forecast_steps: Steps ahead to forecast.
    :param bandwidth: Kernel bandwidth parameter.
    :return: Dictionary containing actual and predicted values.
    """
    predictions = kernel_analog_forecasting(data, lag, forecast_steps, bandwidth)

    # Align actual values with predictions
    actual = data[lag + forecast_steps:]
    predictions = np.array(predictions)

    return  actual,predictions
    