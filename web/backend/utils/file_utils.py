import os
import tempfile
from typing import Optional

def download_temp_file(file_path: str, button_label: str = "📥 Скачать файл") -> None:
    """Скачать временный файл"""
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            file_data = file.read()

        file_name = os.path.basename(file_path)

        import streamlit as st
        st.download_button(
            label=button_label + " " + file_name,
            data=file_data,
            file_name=file_name,
            mime="application/octet-stream",
        )
    else:
        import streamlit as st
        st.error(f"Файл {file_path} не найден")

def load_data_to_tmp(response) -> str:
    """Загрузить данные во временный файл"""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(response)
        return tmp_file.name

def load_model_to_tmp(response) -> str:
    """Загрузить модель во временный файл"""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        tmp_file.write(response)
        return tmp_file.name

def load_python_file_to_tmp(response) -> str:
    """Загрузить Python файл во временный файл"""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp_file:
        tmp_file.write(response)
        return tmp_file.name

def get_loss_func_as_str(filepath: str) -> str:
    """Извлечь функцию потерь из файла как строку"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading loss function: {e}")
        return ""