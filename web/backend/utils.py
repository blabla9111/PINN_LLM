import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta

from .EpidParams.EpidParams import EpidParams
from .PINN_utils.PINN_class import DINN


def load_model(filepath, t, S_data, I_data, D_data, R_data, train_size):
    """Загрузить модель"""
    print("загрузка модели")
    # Определяем девайс: cuda если есть, иначе cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)

    # Создаем экземпляр модели
    model = DINN(t, S_data, I_data, D_data, R_data, device, train_size)

    # Загружаем параметры
    model.load_state_dict(checkpoint["model_state_dict"])
    model.beta_tilda = checkpoint["beta_tilda"]
    model.gamma_tilda = checkpoint["gamma_tilda"]
    model.S_max = checkpoint["S_max"]
    model.I_max = checkpoint["I_max"]
    model.D_max = checkpoint["D_max"]
    model.R_max = checkpoint["R_max"]
    model.S_min = checkpoint["S_min"]
    model.I_min = checkpoint["I_min"]
    model.D_min = checkpoint["D_min"]
    model.R_min = checkpoint["R_min"]
    model.t = checkpoint["t"]
    model.S = checkpoint["S"]
    model.I = checkpoint["I"]
    model.D = checkpoint["D"]
    model.R = checkpoint["R"]

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

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    return metrics


def compare_metrics(
    metrics_dict1, metrics_dict2, model1_name="Модель 1", model2_name="Модель 2"
):
    """
    Создает таблицу для сравнения метрик двух моделей

    Parameters:
    metrics_dict1: dict, метрики первой модели
    metrics_dict2: dict, метрики второй модели
    model1_name: str, название первой модели
    model2_name: str, название второй модели
    """

    # Создаем DataFrame для сравнения
    comparison_df = pd.DataFrame(
        {
            "Метрика": list(metrics_dict1.keys()),
            model1_name: list(metrics_dict1.values()),
            model2_name: list(metrics_dict2.values()),
        }
    )

    # Добавляем разницу между моделями
    comparison_df["Разница"] = comparison_df[model1_name] - comparison_df[model2_name]

    # Форматируем числа для лучшего отображения
    for col in [model1_name, model2_name, "Разница"]:
        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")

    # Отображаем таблицу
    # st.header("📊 Сравнение метрик моделей")
    st.dataframe(comparison_df, hide_index=True, width="stretch")

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
            label=button_label + " " + file_name,
            data=file_data,
            file_name=file_name,
            mime="application/octet-stream",
            on_click="ignore",
        )
    else:
        st.error(f"Файл {file_path} не найден")


def get_data_for_model(filepath):
    x = 180  # сколько дней для обучения

    covid_cases = pd.read_csv(filepath)
    susceptible = []
    infected = []
    dead = []
    recovered = []
    timesteps = []

    d1 = covid_cases["S"]
    d2 = covid_cases["I"]
    d3 = covid_cases["D"]
    d4 = covid_cases["R"]
    d5 = covid_cases["t"]

    for item in range(len(d5)):
        if item % 1 == 0:
            susceptible.append(d1[item])
            infected.append(d2[item])
            dead.append(d3[item])
            recovered.append(d4[item])
            timesteps.append(d5[item])

    return timesteps, susceptible, infected, dead, recovered, x


def translate_to_en(text):
    from deep_translator import GoogleTranslator

    # Используем контекстный менеджер (работает с deep-translator)
    translator = GoogleTranslator(source="ru", target="en")
    translated_text = translator.translate(text)
    # print(translated_text)
    return translated_text


def plot_sidr_predictions(
    timesteps,
    x,
    susceptible,
    infected,
    dead,
    recovered,
    S_pred,
    I_pred,
    D_pred,
    R_pred,
    figsize=(15, 12),
):
    """
    Создает графики предсказаний модели SIDR в формате 2x2

    Parameters:
    -----------
    timesteps : array-like
        Массив временных шагов
    x : int
        Индекс разделения данных на обучающую и тестовую части
    susceptible : array-like
        Реальные данные по восприимчивым
    infected : array-like
        Реальные данные по инфицированным
    deceased : array-like
        Реальные данные по умершим
    recovered : array-like
        Реальные данные по выздоровевшим
    S_pred, I_pred, D_pred, R_pred : tensor
        Предсказания модели для соответствующих групп
    figsize : tuple, optional
        Размер фигуры (по умолчанию (15, 12))

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Объект фигуры с графиками
    """

    # Создаем фигуру с 2x2 субплотов
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # График 1: Susceptible (S) - верхний левый
    ax1.scatter(
        timesteps[:x][::10],
        susceptible[:x][::10],
        c="blue",
        alpha=0.5,
        lw=0.5,
        label="Real data",
    )
    ax1.scatter(
        timesteps[x:][::10],
        susceptible[x:][::10],
        c="white",
        edgecolors="black",
        alpha=0.5,
        lw=0.5,
        label="Future data",
    )
    ax1.plot(
        timesteps,
        S_pred.detach().numpy(),
        "black",
        alpha=0.9,
        lw=2,
        label="Model",
        linestyle="dashed",
    )
    ax1.set_title("Susceptible (S)")
    ax1.set_xlabel("Time, days")
    ax1.set_ylabel("Persons")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Infected (I) - верхний правый
    ax2.scatter(
        timesteps[:x][::10],
        infected[:x][::10],
        c="blue",
        alpha=0.5,
        lw=0.5,
        label="Real data",
    )
    ax2.scatter(
        timesteps[x:][::10],
        infected[x:][::10],
        c="white",
        edgecolors="black",
        alpha=0.5,
        lw=0.5,
        label="Future data",
    )
    ax2.plot(
        timesteps,
        I_pred.detach().numpy(),
        "black",
        alpha=0.9,
        lw=2,
        label="Model",
        linestyle="dashed",
    )
    ax2.set_title("Infected (I)")
    ax2.set_xlabel("Time, days")
    ax2.set_ylabel("Persons")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Recovered (R) - нижний правый
    ax3.scatter(
        timesteps[:x][::10],
        recovered[:x][::10],
        c="blue",
        alpha=0.5,
        lw=0.5,
        label="Real data",
    )
    ax3.scatter(
        timesteps[x:][::10],
        recovered[x:][::10],
        c="white",
        edgecolors="black",
        alpha=0.5,
        lw=0.5,
        label="Future data",
    )
    ax3.plot(
        timesteps,
        R_pred.detach().numpy(),
        "black",
        alpha=0.9,
        lw=2,
        label="Model",
        linestyle="dashed",
    )
    ax3.set_title("Recovered (R)")
    ax3.set_xlabel("Time, days")
    ax3.set_ylabel("Persons")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Deceased (D) - нижний левый
    ax4.scatter(
        timesteps[:x][::10],
        dead[:x][::10],
        c="blue",
        alpha=0.5,
        lw=0.5,
        label="Real data",
    )
    ax4.scatter(
        timesteps[x:][::10],
        dead[x:][::10],
        c="white",
        edgecolors="black",
        alpha=0.5,
        lw=0.5,
        label="Future data",
    )
    ax4.plot(
        timesteps,
        D_pred.detach().numpy(),
        "black",
        alpha=0.9,
        lw=2,
        label="Model",
        linestyle="dashed",
    )
    ax4.set_title("Dead (D)")
    ax4.set_xlabel("Time, days")
    ax4.set_ylabel("Persons")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Настраиваем отступы между графиками
    plt.tight_layout()

    return fig



def plot_sidr_predictions_plotly(
    timesteps,
    x,
    susceptible,
    infected,
    dead,
    recovered,
    S_pred,
    I_pred,
    D_pred,
    R_pred,
    figsize=(500, 300),
    start_day=None
):
    """
    Создает интерактивные графики предсказаний модели SIDR в формате 2x2 с использованием Plotly

    Parameters:
    -----------
    timesteps : array-like
        Массив временных шагов
    x : int
        Индекс разделения данных на обучающую и тестовую части
    susceptible : array-like
        Реальные данные по восприимчивым
    infected : array-like
        Реальные данные по инфицированным
    dead : array-like
        Реальные данные по умершим
    recovered : array-like
        Реальные данные по выздоровевшим
    S_pred, I_pred, D_pred, R_pred : tensor
        Предсказания модели для соответствующих групп
    figsize : tuple, optional
        Размер фигуры в пикселях (по умолчанию (1000, 800))

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Объект фигуры с графиками
    """

    # Конвертируем тензоры в numpy массивы если необходимо
    if hasattr(S_pred, "detach"):
        S_pred = S_pred.detach().numpy()
    if hasattr(I_pred, "detach"):
        I_pred = I_pred.detach().numpy()
    if hasattr(D_pred, "detach"):
        D_pred = D_pred.detach().numpy()
    if hasattr(R_pred, "detach"):
        R_pred = R_pred.detach().numpy()

    # Конвертируем в списки если это тензоры/numpy массивы
    timesteps = list(timesteps) if hasattr(timesteps, "__iter__") else timesteps
    susceptible = list(susceptible) if hasattr(susceptible, "__iter__") else susceptible
    infected = list(infected) if hasattr(infected, "__iter__") else infected
    dead = list(dead) if hasattr(dead, "__iter__") else dead
    recovered = list(recovered) if hasattr(recovered, "__iter__") else recovered

    # Создаем субплоты 2x2
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Susceptible (S)", "Infected (I)", "Dead (D)", "Recovered (R)"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Создаем индексы для каждой 10-й точки
    train_indices = list(range(0, x, 10))
    test_indices = list(range(x, len(timesteps), 10))

    # Функция для выборки данных по индексам
    def get_sampled_data(data, indices):
        return [data[i] for i in indices if i < len(data)]
    
    # Преобразуем временные шаги в даты если указана начальная дата
    if start_day is not None:
        # Конвертируем start_day в datetime объект если это строка
        if isinstance(start_day, str):
            try:
                start_date = datetime.strptime(start_day, '%Y-%m-%d')
            except ValueError:
                start_date = datetime.strptime(start_day, '%d.%m.%Y')
        else:
            start_date = start_day
        
        # Создаем массив дат
        date_labels = [start_date + timedelta(days=int(day)) for day in timesteps]
        x_labels = date_labels
        x_title = "Date"
    else:
        x_labels = timesteps
        x_title = "Time, days"
    
    # Функция для получения x-значений (дат или дней)
    def get_x_values(indices):
        if start_day is not None:
            return [date_labels[i] for i in indices if i < len(date_labels)]
        else:
            return [timesteps[i] for i in indices if i < len(timesteps)]

    # График 1: Susceptible (S) - верхний левый
    fig.add_trace(
        go.Scatter(
            x=get_x_values(train_indices),
            y=get_sampled_data(susceptible, train_indices),
            mode="markers",
            marker=dict(color="blue", size=6, opacity=0.7),
            name="Real data (train)",
            legendgroup="real_train",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=get_x_values(test_indices),
            y=get_sampled_data(susceptible, test_indices),
            mode="markers",
            marker=dict(
                color="white", size=6, opacity=0.7, line=dict(color="black", width=1)
            ),
            name="Future data",
            legendgroup="future",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=S_pred,
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="Model prediction",
            legendgroup="model",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # График 2: Infected (I) - верхний правый
    fig.add_trace(
        go.Scatter(
            x=get_x_values(train_indices),
            y=get_sampled_data(infected, train_indices),
            mode="markers",
            marker=dict(color="blue", size=6, opacity=0.7),
            name="Real data (train)",
            legendgroup="real_train",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=get_x_values(test_indices),
            y=get_sampled_data(infected, test_indices),
            mode="markers",
            marker=dict(
                color="white", size=6, opacity=0.7, line=dict(color="black", width=1)
            ),
            name="Future data",
            legendgroup="future",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=I_pred,
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="Model prediction",
            legendgroup="model",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # График 3: Dead (D) - нижний левый
    fig.add_trace(
        go.Scatter(
            x=get_x_values(train_indices),
            y=get_sampled_data(dead, train_indices),
            mode="markers",
            marker=dict(color="blue", size=6, opacity=0.7),
            name="Real data (train)",
            legendgroup="real_train",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=get_x_values(test_indices),
            y=get_sampled_data(dead, test_indices),
            mode="markers",
            marker=dict(
                color="white", size=6, opacity=0.7, line=dict(color="black", width=1)
            ),
            name="Future data",
            legendgroup="future",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=D_pred,
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="Model prediction",
            legendgroup="model",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # График 4: Recovered (R) - нижний правый
    fig.add_trace(
        go.Scatter(
            x=get_x_values(train_indices),
            y=get_sampled_data(recovered, train_indices),
            mode="markers",
            marker=dict(color="blue", size=6, opacity=0.7),
            name="Real data (train)",
            legendgroup="real_train",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=get_x_values(test_indices),
            y=get_sampled_data(recovered, test_indices),
            mode="markers",
            marker=dict(
                color="white", size=6, opacity=0.7, line=dict(color="black", width=1)
            ),
            name="Future data",
            legendgroup="future",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=R_pred,
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="Model prediction",
            legendgroup="model",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Обновляем layout
    fig.update_layout(
        title_text="SIRD Model Predictions",
        width=figsize[0],
        height=figsize[1],
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Обновляем оси для всех субплотов
    for i in range(1, 5):
        fig.update_xaxes(
            title_text="Time, days", row=(i + 1) // 2, col=2 if i % 2 == 0 else 1
        )
        fig.update_yaxes(
            title_text="Persons", row=(i + 1) // 2, col=2 if i % 2 == 0 else 1
        )

    # Добавляем сетку
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_layout(height=750)

    return fig


def plot_comparison_single_OLD(
    timesteps,
    x,
    real_data,
    pred_old,
    pred_new,
    title,
    ylabel,
    figsize=(700, 500),
    sampling_step=10,
):
    """
    Функция для построения графика сравнения двух моделей для одного компонента

    Parameters:
    -----------
    timesteps : array-like
        Массив временных шагов
    x : int
        Индекс разделения данных на обучающую и тестовую части
    real_data : array-like
        Реальные данные
    pred_old : tensor
        Предсказания старой модели
    pred_new : tensor
        Предсказания новой модели
    title : str
        Заголовок графика
    ylabel : str
        Подпись оси Y
    figsize : tuple, optional
        Размер фигуры (по умолчанию (10, 6))
    sampling_step : int, optional
        Шаг дискретизации для scatter plot (по умолчанию 10)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Объект фигуры с графиком
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        timesteps[:x][::sampling_step],
        real_data[:x][::sampling_step],
        c="blue",
        alpha=0.5,
        lw=0.5,
        label="Real data",
    )

    ax.scatter(
        timesteps[x:][::sampling_step],
        real_data[x:][::sampling_step],
        c="white",
        edgecolors="black",
        alpha=0.5,
        lw=0.5,
        label="Future data",
    )

    ax.plot(
        timesteps,
        pred_old.detach().numpy(),
        "black",
        alpha=0.9,
        lw=2,
        label="Old Model",
        linestyle="dashed",
    )
    ax.plot(
        timesteps,
        pred_new.detach().numpy(),
        "red",
        alpha=0.9,
        lw=2,
        label="New Model",
        linestyle="dashed",
    )

    ax.set_xlabel("Time, days")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def plot_comparison_single(
    timesteps,
    x,
    real_data,
    pred_old,
    pred_new,
    title,
    ylabel,
    sampling_step=10
):
    """
    Plot comparison of two models for one component using Plotly

    Parameters:
    -----------
    timesteps : array-like
        Array of time steps
    x : int
        Index splitting train and test data parts
    real_data : array-like
        Real data
    pred_old : array-like
        Predictions of old model
    pred_new : array-like
        Predictions of new model
    title : str
        Plot title
    ylabel : str
        Y-axis label
    sampling_step : int, optional
        Sampling step for scatter plot (default 10)

    Returns:
    --------
    fig : plotly.graph_objs.Figure
        Plotly Figure object
    """

    # Создаем индексы для каждой 10-й точки
    train_indices = list(range(0, x, 10))
    test_indices = list(range(x, len(timesteps), 10))

    # Функция для выборки данных по индексам
    def get_sampled_data(data, indices):
        return [data[i] for i in indices if i < len(data)]

    fig = go.Figure()


    # Scatter real data up to index x
    fig.add_trace(go.Scatter(
        x=get_sampled_data(timesteps, train_indices),
        y=get_sampled_data(real_data, train_indices),
        mode='markers',
        marker=dict(color='blue', opacity=0.5, size=6, line=dict(width=0.5)),
        name='Real data'
    ))

    # Scatter real future data after index x
    fig.add_trace(go.Scatter(
        x=get_sampled_data(timesteps, test_indices),
        y=get_sampled_data(real_data, test_indices),
        mode='markers',
        marker=dict(color='white', opacity=0.5, size=6, line=dict(color='black', width=0.5)),
        name='Future data'
    ))

    # Line for old model prediction
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=pred_old,
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Old Model'
    ))

    # Line for new model prediction
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=pred_new,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='New Model'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time, days',
        yaxis_title=ylabel,
        legend=dict(x=0, y=1),
        template='plotly_white'
    )
    return fig

# Функции для отдельных компонентов
def plot_S_comparison(
    timesteps, x, susceptible, S_pred_old, S_pred_new, figsize=(10, 6)
):
    """График сравнения для Susceptible"""
    return plot_comparison_single(
        timesteps,
        x,
        susceptible,
        S_pred_old,
        S_pred_new,
        "Susceptible (S) Comparison",
        "Susceptible, persons",
        figsize,
    )


def plot_I_comparison(timesteps, x, infected, I_pred_old, I_pred_new, figsize=(10, 6)):
    """График сравнения для Infected"""
    return plot_comparison_single(
        timesteps,
        x,
        infected,
        I_pred_old,
        I_pred_new,
        "Infected (I) Comparison",
        "Infected, persons",
        figsize,
    )


def plot_R_comparison(timesteps, x, recovered, R_pred_old, R_pred_new, figsize=(10, 6)):
    """График сравнения для Recovered"""
    return plot_comparison_single(
        timesteps,
        x,
        recovered,
        R_pred_old,
        R_pred_new,
        "Recovered (R) Comparison",
        "Recovered, persons",
        figsize,
    )


def plot_D_comparison(timesteps, x, deceased, D_pred_old, D_pred_new, figsize=(10, 6)):
    """График сравнения для Deceased"""
    return plot_comparison_single(
        timesteps,
        x,
        deceased,
        D_pred_old,
        D_pred_new,
        "Dead (D) Comparison",
        "Dead, persons",
        figsize,
    )


def get_R0(S, I, R, D, timesteps):
    epidParams = EpidParams(S, I, R, D, timesteps)
    return epidParams.R0()


def get_Rt_array(S, I, R, D, timesteps):
    epidParams = EpidParams(S, I, R, D, timesteps)
    return epidParams.Rt_array()


def get_Rt(S, I, R, D, timesteps, t):
    epidParams = EpidParams(S, I, R, D, timesteps)
    return epidParams.Rt(t)


def display_epid_params(S_pred, I_pred, R_pred, D_pred, timesteps):
    # дне не соответствуют в отображении
    Rt_array = get_Rt_array(S_pred, I_pred, R_pred, D_pred, timesteps)

    # Создание интерактивного графика
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=Rt_array,
            mode="lines+markers",  # линии + точки
            name="Временной ряд",
            line=dict(color="blue", width=2),
            marker=dict(size=4, color="red"),
            hovertemplate="<b>Время:</b> %{x:.2f}<br><b>Значение:</b> %{y:.4f}<extra></extra>",
        )
    )

    # Настройка внешнего вида
    fig.update_layout(
        title="Временной ряд Rt (effective reproduction number)",
        xaxis_title="Время (t)",
        yaxis_title="Значение (Rt)",
        hovermode="x unified",  # показывает все точки на одной вертикали
        template="plotly_white",
        height=500,
    )

    return fig


def display_compared_epid_params(
    S_pred_1,
    I_pred_1,
    R_pred_1,
    D_pred_1,
    timesteps_1,
    S_pred_2,
    I_pred_2,
    R_pred_2,
    D_pred_2,
    timesteps_2,
):
    # дни не соответствуют в отображении
    Rt_array_1 = get_Rt_array(S_pred_1, I_pred_1, R_pred_1, D_pred_1, timesteps_1)
    Rt_array_2 = get_Rt_array(S_pred_2, I_pred_2, R_pred_2, D_pred_2, timesteps_2)

    # Создание интерактивного графика с двумя рядами
    fig = go.Figure()

    # Первый временной ряд
    fig.add_trace(
        go.Scatter(
            x=timesteps_1,
            y=Rt_array_1,
            mode="lines+markers",
            name="PINN",
            line=dict(color="blue", width=2),
            marker=dict(size=4, color="red"),
            hovertemplate="<b>Время:</b> %{x:.2f}<br><b>Значение:</b> %{y:.4f}<extra></extra>",
        )
    )

    # Второй временной ряд
    fig.add_trace(
        go.Scatter(
            x=timesteps_2,
            y=Rt_array_2,
            mode="lines+markers",
            name="NEW_PINN",
            line=dict(color="green", width=2),
            marker=dict(size=4, color="orange"),
            hovertemplate="<b>Время:</b> %{x:.2f}<br><b>Значение:</b> %{y:.4f}<extra></extra>",
        )
    )

    # Настройка внешнего вида
    fig.update_layout(
        title="Сравнение Rt_array",
        xaxis_title="Время (t)",
        yaxis_title="Значение (Rt_array)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig

import streamlit.components.v1 as components

def show_mode_indicator():
    if 'mode' in st.session_state:
        if st.session_state.mode == 'DEV':
            footer_css = """
                <style>
                .footer {
                    position: fixed;
                    left: 0;
                    bottom: 0;
                    width: 100%;
                    background-color: #FF4B4B;
                    font-weight: bold;
                    font-family: sans-serif;
                    color: black;
                    text-align: center;
                    padding: 10px;
                    border-top: 1px solid #ddd;
                    z-index: 1000;
                }
                </style>

                <div class="footer">
                    🛠️ Включен режим разработчика
                </div>
                """

            components.html(footer_css, height=50)
