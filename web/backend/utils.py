from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from web.backend.PINN_utils.PINN_class import DINN
import numpy as np
import pandas as pd
import streamlit as st
import os

def load_model(filepath, t, S_data, I_data, D_data, R_data):
    """Загрузить модель"""
    print("загрузка модели")
    checkpoint = torch.load(filepath)

    # Создаем экземпляр модели
    model = DINN(t, S_data, I_data, D_data, R_data)

    # Загружаем параметры
    model.load_state_dict(checkpoint['model_state_dict'])
    model.beta_tilda = checkpoint['beta_tilda']
    model.gamma_tilda = checkpoint['gamma_tilda']
    model.S_max = checkpoint['S_max']
    model.I_max = checkpoint['I_max']
    model.D_max = checkpoint['D_max']
    model.R_max = checkpoint['R_max']
    model.S_min = checkpoint['S_min']
    model.I_min = checkpoint['I_min']
    model.D_min = checkpoint['D_min']
    model.R_min = checkpoint['R_min']
    model.t = checkpoint['t']
    model.S = checkpoint['S']
    model.I = checkpoint['I']
    model.D = checkpoint['D']
    model.R = checkpoint['R']

    # Обновляем производные атрибуты
    model.t_float = model.t.float()
    model.t_batch = torch.reshape(model.t_float, (len(model.t), 1))
    model.S_hat = (model.S - model.S_min) / (model.S_max - model.S_min)
    model.I_hat = (model.I - model.I_min) / (model.I_max - model.I_min)
    model.D_hat = (model.D - model.D_min) / (model.D_max - model.D_min)
    model.R_hat = (model.R - model.R_min) / (model.R_max - model.R_min)

    print(f"Model loaded from {filepath}")
    return model


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

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

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
        'Метрика': list(metrics_dict1.keys()),
        model1_name: list(metrics_dict1.values()),
        model2_name: list(metrics_dict2.values())
    })

    # Добавляем разницу между моделями
    comparison_df['Разница'] = comparison_df[model1_name] - \
        comparison_df[model2_name]

    # Форматируем числа для лучшего отображения
    for col in [model1_name, model2_name, 'Разница']:
        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")

    # Отображаем таблицу
    st.header("📊 Сравнение метрик моделей")
    st.dataframe(comparison_df, hide_index=True, width='stretch')

    return comparison_df


def download_temp_file(file_path, button_label="📥 Скачать файл"):
    """
    Функция для скачивания временного файла
    
    file_path: путь к временному файлу
    button_label: текст на кнопке скачивания
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            file_data = file.read()

        file_name = os.path.basename(file_path)

        st.download_button(
            label=button_label+" "+file_name,
            data=file_data,
            file_name=file_name,
            mime="application/octet-stream",
            on_click="ignore"
        )
    else:
        st.error(f"Файл {file_path} не найден")

def get_data_for_model(filepath):
    x = 180 # сколько дней для обучения

    covid_cases = pd.read_csv(filepath)
    susceptible = []
    infected = []
    dead = []
    recovered = []
    timesteps = []

    d1 = covid_cases['S']
    d2 = covid_cases['I']
    d3 = covid_cases['D']
    d4 = covid_cases['R']
    d5 = covid_cases['t']

    for item in range(len(d5)):
        if item % 1 == 0:
            susceptible.append(d1[item])
            infected.append(d2[item])
            dead.append(d3[item])
            recovered.append(d4[item])
            timesteps.append(d5[item])


    return timesteps, susceptible, infected, dead, recovered, x
