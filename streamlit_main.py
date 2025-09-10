import streamlit as st
from web.frontend import pages as pg

# Настройка страницы
st.set_page_config(page_title="Анализ модели", layout="wide")

# Инициализация session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

# Отображение текущей страницы
if st.session_state.current_page == "main":
    pg.start_page()
elif st.session_state.current_page == "results":
    pg.class_subclass_page()
elif st.session_state.current_page == "generate new model":
    pg.generate_model_page()

# Стилизация
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
    .stInfo {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1890ff;
    }
</style>
""", unsafe_allow_html=True)
