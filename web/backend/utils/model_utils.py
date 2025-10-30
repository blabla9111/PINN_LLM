import torch
import streamlit as st
import streamlit.components.v1 as components
from deep_translator import GoogleTranslator

from ..PINN_utils.PINN_class import DINN

def load_model(filepath, t, S_data, I_data, D_data, R_data, train_size):
    """Загрузить модель DINN"""
    print("загрузка модели")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)

    model = DINN(t, S_data, I_data, D_data, R_data, device, train_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Загружаем параметры
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

def translate_to_en(text: str) -> str:
    """Перевести текст с русского на английский"""
    translator = GoogleTranslator(source="ru", target="en")
    return translator.translate(text)

def show_mode_indicator():
    """Показать индикатор режима разработчика"""
    if 'mode' in st.session_state and st.session_state.mode == 'DEV':
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