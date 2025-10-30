import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def plot_sidr_predictions(
    timesteps, x, susceptible, infected, dead, recovered, 
    S_pred, I_pred, D_pred, R_pred, figsize=(15, 12)
):
    """Создает графики предсказаний модели SIDR в формате 2x2 (matplotlib)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # График 1: Susceptible (S)
    ax1.scatter(timesteps[:x][::10], susceptible[:x][::10], c="blue", alpha=0.5, lw=0.5, label="Real data")
    ax1.scatter(timesteps[x:][::10], susceptible[x:][::10], c="white", edgecolors="black", alpha=0.5, lw=0.5, label="Future data")
    ax1.plot(timesteps, S_pred.detach().numpy(), "black", alpha=0.9, lw=2, label="Model", linestyle="dashed")
    ax1.set_title("Susceptible (S)")
    ax1.set_xlabel("Time, days")
    ax1.set_ylabel("Persons")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Infected (I)
    ax2.scatter(timesteps[:x][::10], infected[:x][::10], c="blue", alpha=0.5, lw=0.5, label="Real data")
    ax2.scatter(timesteps[x:][::10], infected[x:][::10], c="white", edgecolors="black", alpha=0.5, lw=0.5, label="Future data")
    ax2.plot(timesteps, I_pred.detach().numpy(), "black", alpha=0.9, lw=2, label="Model", linestyle="dashed")
    ax2.set_title("Infected (I)")
    ax2.set_xlabel("Time, days")
    ax2.set_ylabel("Persons")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Recovered (R)
    ax3.scatter(timesteps[:x][::10], recovered[:x][::10], c="blue", alpha=0.5, lw=0.5, label="Real data")
    ax3.scatter(timesteps[x:][::10], recovered[x:][::10], c="white", edgecolors="black", alpha=0.5, lw=0.5, label="Future data")
    ax3.plot(timesteps, R_pred.detach().numpy(), "black", alpha=0.9, lw=2, label="Model", linestyle="dashed")
    ax3.set_title("Recovered (R)")
    ax3.set_xlabel("Time, days")
    ax3.set_ylabel("Persons")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Dead (D)
    ax4.scatter(timesteps[:x][::10], dead[:x][::10], c="blue", alpha=0.5, lw=0.5, label="Real data")
    ax4.scatter(timesteps[x:][::10], dead[x:][::10], c="white", edgecolors="black", alpha=0.5, lw=0.5, label="Future data")
    ax4.plot(timesteps, D_pred.detach().numpy(), "black", alpha=0.9, lw=2, label="Model", linestyle="dashed")
    ax4.set_title("Dead (D)")
    ax4.set_xlabel("Time, days")
    ax4.set_ylabel("Persons")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_sidr_predictions_plotly(
    timesteps, x, susceptible, infected, dead, recovered,
    S_pred, I_pred, D_pred, R_pred, figsize=(500, 300), start_day=None
):
    """Создает интерактивные графики предсказаний модели SIDR в формате 2x2 с использованием Plotly"""
    
    # Конвертируем тензоры в numpy массивы если необходимо
    if hasattr(S_pred, "detach"): S_pred = S_pred.detach().numpy()
    if hasattr(I_pred, "detach"): I_pred = I_pred.detach().numpy()
    if hasattr(D_pred, "detach"): D_pred = D_pred.detach().numpy()
    if hasattr(R_pred, "detach"): R_pred = R_pred.detach().numpy()

    # Конвертируем в списки если это тензоры/numpy массивы
    timesteps = list(timesteps) if hasattr(timesteps, "__iter__") else timesteps
    susceptible = list(susceptible) if hasattr(susceptible, "__iter__") else susceptible
    infected = list(infected) if hasattr(infected, "__iter__") else infected
    dead = list(dead) if hasattr(dead, "__iter__") else dead
    recovered = list(recovered) if hasattr(recovered, "__iter__") else recovered

    # Создаем субплоты 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Susceptible (S)", "Infected (I)", "Dead (D)", "Recovered (R)"),
        vertical_spacing=0.15, horizontal_spacing=0.1,
    )

    # Создаем индексы для каждой 10-й точки
    train_indices = list(range(0, x, 10))
    test_indices = list(range(x, len(timesteps), 10))

    def get_sampled_data(data, indices):
        return [data[i] for i in indices if i < len(data)]
    
    # Преобразуем временные шаги в даты если указана начальная дата
    if start_day is not None:
        if isinstance(start_day, str):
            try:
                start_date = datetime.strptime(start_day, '%Y-%m-%d')
            except ValueError:
                start_date = datetime.strptime(start_day, '%d.%m.%Y')
        else:
            start_date = start_day
        
        date_labels = [start_date + timedelta(days=int(day)) for day in timesteps]
        x_labels = date_labels
    else:
        x_labels = timesteps
    
    def get_x_values(indices):
        if start_day is not None:
            return [date_labels[i] for i in indices if i < len(date_labels)]
        else:
            return [timesteps[i] for i in indices if i < len(timesteps)]

    # График 1: Susceptible (S)
    fig.add_trace(go.Scatter(x=get_x_values(train_indices), y=get_sampled_data(susceptible, train_indices),
                           mode="markers", marker=dict(color="blue", size=6, opacity=0.7), name="Real data (train)", legendgroup="real_train", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=get_x_values(test_indices), y=get_sampled_data(susceptible, test_indices),
                           mode="markers", marker=dict(color="white", size=6, opacity=0.7, line=dict(color="black", width=1)), name="Future data", legendgroup="future", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=S_pred, mode="lines", line=dict(color="black", width=3, dash="dash"),
                           name="Model prediction", legendgroup="model", showlegend=True), row=1, col=1)

    # График 2: Infected (I)
    fig.add_trace(go.Scatter(x=get_x_values(train_indices), y=get_sampled_data(infected, train_indices),
                           mode="markers", marker=dict(color="blue", size=6, opacity=0.7), name="Real data (train)", legendgroup="real_train", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=get_x_values(test_indices), y=get_sampled_data(infected, test_indices),
                           mode="markers", marker=dict(color="white", size=6, opacity=0.7, line=dict(color="black", width=1)), name="Future data", legendgroup="future", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_labels, y=I_pred, mode="lines", line=dict(color="black", width=3, dash="dash"),
                           name="Model prediction", legendgroup="model", showlegend=False), row=1, col=2)

    # График 3: Dead (D)
    fig.add_trace(go.Scatter(x=get_x_values(train_indices), y=get_sampled_data(dead, train_indices),
                           mode="markers", marker=dict(color="blue", size=6, opacity=0.7), name="Real data (train)", legendgroup="real_train", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=get_x_values(test_indices), y=get_sampled_data(dead, test_indices),
                           mode="markers", marker=dict(color="white", size=6, opacity=0.7, line=dict(color="black", width=1)), name="Future data", legendgroup="future", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=D_pred, mode="lines", line=dict(color="black", width=3, dash="dash"),
                           name="Model prediction", legendgroup="model", showlegend=False), row=2, col=1)

    # График 4: Recovered (R)
    fig.add_trace(go.Scatter(x=get_x_values(train_indices), y=get_sampled_data(recovered, train_indices),
                           mode="markers", marker=dict(color="blue", size=6, opacity=0.7), name="Real data (train)", legendgroup="real_train", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=get_x_values(test_indices), y=get_sampled_data(recovered, test_indices),
                           mode="markers", marker=dict(color="white", size=6, opacity=0.7, line=dict(color="black", width=1)), name="Future data", legendgroup="future", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_labels, y=R_pred, mode="lines", line=dict(color="black", width=3, dash="dash"),
                           name="Model prediction", legendgroup="model", showlegend=False), row=2, col=2)

    # Обновляем layout
    fig.update_layout(title_text="SIRD Model Predictions", width=figsize[0], height=figsize[1],
                     template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Обновляем оси
    for i in range(1, 5):
        fig.update_xaxes(title_text="Time, days", row=(i + 1) // 2, col=2 if i % 2 == 0 else 1)
        fig.update_yaxes(title_text="Persons", row=(i + 1) // 2, col=2 if i % 2 == 0 else 1)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_layout(height=750)

    return fig

def plot_comparison_single(timesteps, x, real_data, pred_old, pred_new, title, ylabel, sampling_step=10):
    """Plot comparison of two models for one component using Plotly"""
    
    # Создаем индексы для каждой 10-й точки
    train_indices = list(range(0, x, 10))
    test_indices = list(range(x, len(timesteps), 10))

    def get_sampled_data(data, indices):
        return [data[i] for i in indices if i < len(data)]

    fig = go.Figure()

    # Scatter real data up to index x
    fig.add_trace(go.Scatter(x=get_sampled_data(timesteps, train_indices), y=get_sampled_data(real_data, train_indices),
                           mode='markers', marker=dict(color='blue', opacity=0.5, size=6, line=dict(width=0.5)), name='Real data'))

    # Scatter real future data after index x
    fig.add_trace(go.Scatter(x=get_sampled_data(timesteps, test_indices), y=get_sampled_data(real_data, test_indices),
                           mode='markers', marker=dict(color='white', opacity=0.5, size=6, line=dict(color='black', width=0.5)), name='Future data'))

    # Line for old model prediction
    fig.add_trace(go.Scatter(x=timesteps, y=pred_old, mode='lines', line=dict(color='black', width=2, dash='dash'), name='Old Model'))

    # Line for new model prediction
    fig.add_trace(go.Scatter(x=timesteps, y=pred_new, mode='lines', line=dict(color='red', width=2, dash='dash'), name='New Model'))

    fig.update_layout(title=title, xaxis_title='Time, days', yaxis_title=ylabel, legend=dict(x=0, y=1), template='plotly_white')
    return fig

# Функции для отдельных компонентов
def plot_S_comparison(timesteps, x, susceptible, S_pred_old, S_pred_new, figsize=(10, 6)):
    """График сравнения для Susceptible"""
    return plot_comparison_single(timesteps, x, susceptible, S_pred_old, S_pred_new, "Susceptible (S) Comparison", "Susceptible, persons", figsize)

def plot_I_comparison(timesteps, x, infected, I_pred_old, I_pred_new, figsize=(10, 6)):
    """График сравнения для Infected"""
    return plot_comparison_single(timesteps, x, infected, I_pred_old, I_pred_new, "Infected (I) Comparison", "Infected, persons", figsize)

def plot_R_comparison(timesteps, x, recovered, R_pred_old, R_pred_new, figsize=(10, 6)):
    """График сравнения для Recovered"""
    return plot_comparison_single(timesteps, x, recovered, R_pred_old, R_pred_new, "Recovered (R) Comparison", "Recovered, persons", figsize)

def plot_D_comparison(timesteps, x, deceased, D_pred_old, D_pred_new, figsize=(10, 6)):
    """График сравнения для Deceased"""
    return plot_comparison_single(timesteps, x, deceased, D_pred_old, D_pred_new, "Dead (D) Comparison", "Dead, persons", figsize)