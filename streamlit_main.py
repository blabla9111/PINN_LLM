import streamlit as st
from web.frontend import pages as pg
from web.backend.database.SupabaseEngine import SupabaseEngine

# Импорт конфигурации
from web.backend.config.config_utils import get_config

# Инициализация конфигурации
config = get_config()

# Настройка страницы
st.set_page_config(
    page_title=config.app.PAGE_TITLE, 
    layout=config.app.PAGE_LAYOUT
)

# Инициализация session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

if "supabase" not in st.session_state:
    # Используем конфиг для инициализации Supabase
    st.session_state['supabase'] = SupabaseEngine(config).supabase

# Добавляем конфиг в session state для доступа на всех страницах
if 'app_config' not in st.session_state:
    st.session_state['app_config'] = config

# Отображение текущей страницы
if st.session_state.current_page == "main":
    pg.start_page()
elif st.session_state.current_page == "results":
    pg.class_subclass_page()
elif st.session_state.current_page == "generate new model":
    pg.generate_model_page()

# Информация о конфигурации в режиме разработчика
if st.session_state.get('mode') == 'DEV':
    with st.sidebar.expander("🔧 Конфигурация"):
        st.write(f"**Режим:** {config.app.MODE}")
        st.write(f"**LLM Модель:** {config.llm.MODEL_NAME}")
        st.write(f"**TensorBoard:** {'✅' if config.training.ENABLE_TENSORBOARD_LOGGING else '❌'}")
        st.write(f"**Bucket:** {config.supabase.STORAGE_BUCKET}")

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

st.markdown("""
    <style>
        .col1-style {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 20px;
            border-radius: 10px;
        }
        .col2-style {
            background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%);
            padding: 20px;
            border-radius: 10px;
        }
        .col3-style {
            background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
            padding: 20px;
            border-radius: 10px;
        }
        .col4-style {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)