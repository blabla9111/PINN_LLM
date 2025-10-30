import plotly.graph_objects as go
from ..EpidParams.EpidParams import EpidParams

def get_R0(S, I, R, D, timesteps):
    """Calculate basic reproduction number R0"""
    epidParams = EpidParams(S, I, R, D, timesteps)
    return epidParams.R0()

def get_Rt_array(S, I, R, D, timesteps):
    """Calculate effective reproduction number array Rt"""
    epidParams = EpidParams(S, I, R, D, timesteps)
    return epidParams.Rt_array()

def get_Rt(S, I, R, D, timesteps, t):
    """Calculate effective reproduction number at time t"""
    epidParams = EpidParams(S, I, R, D, timesteps)
    return epidParams.Rt(t)

def display_epid_params(S_pred, I_pred, R_pred, D_pred, timesteps):
    """Display epidemic parameters (Rt array)"""
    Rt_array = get_Rt_array(S_pred, I_pred, R_pred, D_pred, timesteps)

    # Создание интерактивного графика
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=Rt_array,
            mode="lines+markers",
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
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig

def display_compared_epid_params(
    S_pred_1, I_pred_1, R_pred_1, D_pred_1, timesteps_1,
    S_pred_2, I_pred_2, R_pred_2, D_pred_2, timesteps_2
):
    """Display compared epidemic parameters for two models"""
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