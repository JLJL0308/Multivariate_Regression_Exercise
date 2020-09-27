from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data = boston_dataset.data, columns = boston_dataset.feature_names)
features = data.drop(["INDUS", "AGE"], axis = 1)

# Log prices
log_prices = np.log(boston_dataset.target)
# Covert to 2D array.
target = pd.DataFrame(log_prices, columns = ["PRICE"])

# Create property_stats varible to store
CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8
property_stats = features.mean().values.reshape(1, 11)
ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

regr = LinearRegression().fit(features, target)
fitted_values = regr.predict(features)
MSE = mean_squared_error(target, fitted_values)
RMSE = np.sqrt(MSE)

def get_log_estimate(room_number, pupil_teacher_ratio, beside_river = False, high_confidence = True):
    # Configure property values
    property_stats[0][RM_IDX] = room_number
    property_stats[0][PTRATIO_IDX] = pupil_teacher_ratio
    if beside_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    # Predict
    log_estimate = regr.predict(property_stats)[0][0]
    # Calculate the range
    if high_confidence:
        # 2 standard deviation
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        # 1 standard deviation
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    return log_estimate, upper_bound, lower_bound, interval

def get_today_dollar_estimate(room_number, pupil_teacher_ratio, beside_river = False, high_confidence = True):
    """ Estimate today's property price in Boston.
    """
    if room_number < 1 or pupil_teacher_ratio < 1:
        return "Information is not valid."
    log_estimate, upper_bound, lower_bound, confidence = get_log_estimate(room_number, pupil_teacher_ratio, beside_river, high_confidence)
    estimate_today = np.around(np.e**log_estimate * SCALE_FACTOR * 1000, -3)
    estimate_high = np.around(np.e**upper_bound * SCALE_FACTOR * 1000, -3)
    estimate_low = np.around(np.e**lower_bound * SCALE_FACTOR * 1000, -3)
    print(f"Today's estimate: {estimate_today}, Upper Bound: {estimate_high}, Lower Bound: {estimate_low}, Confidence: {confidence}%")