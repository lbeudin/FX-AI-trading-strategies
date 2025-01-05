# Kalman Filter implementation 
# Léa Beudin using my friend chat gpt
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_state, initial_uncertainty):
        """
        Initialize the Kalman Filter.
        :param process_variance: Variance of the process noise.
        :param measurement_variance: Variance of the measurement noise.
        :param initial_state: Initial estimate of the state.
        :param initial_uncertainty: Initial uncertainty of the state estimate.
        """
        self.process_variance = process_variance  # Q: Process variance
        self.measurement_variance = measurement_variance  # R: Measurement variance
        self.state_estimate = initial_state  # x: Initial state estimate
        self.uncertainty = initial_uncertainty  # P: Initial uncertainty

    def update(self, measurement):
        """
        Update step of the Kalman Filter.
        :param measurement: Observed measurement at the current time step.
        """
        # Kalman Gain
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)

        # Update state estimate and uncertainty
        self.state_estimate = self.state_estimate + kalman_gain * (measurement - self.state_estimate)
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

    def predict(self):
        """
        Prediction step of the Kalman Filter.
        """
        # Increase uncertainty due to process noise
        self.uncertainty += self.process_variance



# Backtesting function
def kalman_filter(data, process_variance, measurement_variance):
    """
    Backtest the Kalman Filter on a time series.
    :param data: Time series data.
    :param process_variance: Variance of the process noise.
    :param measurement_variance: Variance of the measurement noise.
    :return: Dictionary containing actual, predicted values, and filtered values.
    """
    # Initialize Kalman Filter
    initial_state = data[0]
    initial_uncertainty = 1.0
    kf = KalmanFilter(process_variance, measurement_variance, initial_state, initial_uncertainty)

    predictions = []
    filtered_values = []

    # Apply Kalman Filter to the time series
    for measurement in data:
        kf.predict()  # Predict next state
        predictions.append(kf.state_estimate)  # Save predicted state
        kf.update(measurement)  # Update state with current measurement
        filtered_values.append(kf.state_estimate)  # Save filtered state

    return {
        "actual": data,
        "predicted": predictions,
        "filtered": filtered_values
    }