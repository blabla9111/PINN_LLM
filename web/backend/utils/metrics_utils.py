import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics

    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values

    Returns:
    dict: Dictionary with MAE, MSE, RMSE, R2 metrics
    """
    # Convert to numpy arrays if they are tensors
    if torch.is_tensor(y_true):
        y_true = y_true.detach().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().numpy()

    # Ensure they are 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    return metrics

def compare_metrics(metrics_dict1, metrics_dict2, model1_name="Модель 1", model2_name="Модель 2"):
    """
    Создает таблицу для сравнения метрик двух моделей

    Parameters:
    metrics_dict1: dict, метрики первой модели
    metrics_dict2: dict, метрики второй модели
    model1_name: str, название первой модели
    model2_name: str, название второй модели
    """

    # Создаем DataFrame для сравнения
    comparison_df = pd.DataFrame({
        "Метрика": list(metrics_dict1.keys()),
        model1_name: list(metrics_dict1.values()),
        model2_name: list(metrics_dict2.values()),
    })

    # Добавляем разницу между моделями
    comparison_df["Разница"] = comparison_df[model1_name] - comparison_df[model2_name]

    # Форматируем числа для лучшего отображения
    for col in [model1_name, model2_name, "Разница"]:
        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")

    return comparison_df