"""
Evaluation Metrics for Quantitative Financial Sequence Forecasting.
Calculates RMSE, MAE, MAPE, and Directional Hit Rate (%).
"""

import math

def calculate_rmse(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
    return math.sqrt(mse)

def calculate_mae(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

def calculate_mape(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    valid_terms = [abs((yt - yp) / yt) for yt, yp in zip(y_true, y_pred) if yt != 0]
    return (sum(valid_terms) / len(valid_terms)) * 100.0 if valid_terms else 0.0

def calculate_directional_accuracy(y_true, y_pred):
    """
    Hit Rate (%): Measures how often the predicted direction (up/down) matches actual movement.
    """
    if len(y_true) < 2:
        return 50.0
    correct = 0
    total = len(y_true) - 1
    for i in range(1, len(y_true)):
        actual_direction = 1 if y_true[i] >= y_true[i-1] else -1
        pred_direction = 1 if y_pred[i] >= y_true[i-1] else -1
        if actual_direction == pred_direction:
            correct += 1
    return (correct / total) * 100.0

def evaluate_model_predictions(y_true, y_pred):
    acc = round(calculate_directional_accuracy(y_true, y_pred), 2)
    return {
        "RMSE": round(calculate_rmse(y_true, y_pred), 4),
        "MAE": round(calculate_mae(y_true, y_pred), 4),
        "MAPE": round(calculate_mape(y_true, y_pred), 2),
        "Directional_Accuracy": acc,
        "Directional_Accuracy_percent": acc
    }

class TimeSeriesEvaluator:
    rmse = staticmethod(calculate_rmse)
    mae = staticmethod(calculate_mae)
    mape = staticmethod(calculate_mape)
    directional_accuracy = staticmethod(calculate_directional_accuracy)
    evaluate_all = staticmethod(evaluate_model_predictions)
