import tempfile
import os
from pathlib import Path
from typing import Tuple


def create_file_in_tmp(loss_function_str: str, py_file_path: str, start_file_path: str, end_file_path: str) -> Tuple[str, str]:
    """
    Создает Python файл с функцией потерь для PINN, используя начало и конец из файлов
    
    Args:
        loss_function_str (str): код функции потерь
        py_file_path (str): имя создаваемого файла
        start_file_path (str): путь к файлу с начальным кодом
        end_file_path (str): путь к файлу с завершающим кодом
        
    Returns:
        Tuple[str, str]: путь к временному файлу и его содержимое
        
    Raises:
        FileNotFoundError: если не найден один из файлов
        Exception: при ошибках создания файла
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_file_path = temp_dir / py_file_path

    try:
        # Читаем начало и конец из файлов
        start_code = Path(start_file_path).read_text(encoding='utf-8')
        end_code = Path(end_file_path).read_text(encoding='utf-8')

        # Формируем полный код с красивым форматированием
        full_code = f"""{start_code}


{loss_function_str}


{end_code}"""

        # Создаем и записываем файл
        temp_file_path.write_text(full_code, encoding='utf-8')
        
        # Возвращаем путь и содержимое
        return str(temp_file_path), full_code

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл не найден: {e}")
    except Exception as e:
        raise Exception(f"Ошибка при создании файла: {e}")


def load_model_to_tmp(model: bytes) -> str:
    """
    Сохраняет модель во временный файл
    
    Args:
        model (bytes): бинарные данные модели
        
    Returns:
        str: путь к временному файлу
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        tmp_file.write(model)
        tmp_path = tmp_file.name

    print(f"📁 Модель сохранена во временное место: {tmp_path}")
    return tmp_path


def load_data_to_tmp(data_file: bytes) -> str:
    """
    Сохраняет данные во временный CSV файл
    
    Args:
        data_file (bytes): бинарные данные файла
        
    Returns:
        str: путь к временному файлу
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(data_file)
        tmp_path = tmp_file.name

    print(f"📊 Данные сохранены во временное место: {tmp_path}")
    return tmp_path


def load_python_file_to_tmp(python_file: bytes) -> str:
    """
    Сохраняет Python файл во временное расположение
    
    Args:
        python_file (bytes): бинарные данные Python файла
        
    Returns:
        str: путь к временному файлу
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
        tmp_file.write(python_file)
        tmp_path = tmp_file.name

    print(f"🐍 Python файл сохранен во временное место: {tmp_path}")
    return tmp_path