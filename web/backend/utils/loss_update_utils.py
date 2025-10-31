import tempfile
import os
from pathlib import Path
from typing import Tuple


def save(py_file_path: str, code: str, add_imports: bool = True, 
         history_file_path: str = 'results/saved_losses/losses.py') -> str:
    """
    Сохраняет код функции потерь в файл и добавляет в историю
    
    Args:
        py_file_path (str): путь к файлу для сохранения
        code (str): код функции потерь
        add_imports (bool): добавлять ли импорт torch
        history_file_path (str): путь к файлу истории
        
    Returns:
        str: путь к сохраненному файлу
    """
    # Добавляем необходимые импорты
    if add_imports:
        code = "\n\nimport torch\n\n\n" + code

    # Сохраняем в основной файл
    with open(py_file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Сохраняем в историю
    save_to_history(history_file_path, code)
    
    return py_file_path


def save_py(py_file_path: str, code: str, add_imports: bool = True, 
            history_file_path: str = 'results/saved_losses/losses.py') -> Tuple[str, str]:
    """
    Сохраняет код функции потерь во временный файл и добавляет в историю
    
    Args:
        py_file_path (str): имя создаваемого файла
        code (str): код функции потерь
        add_imports (bool): добавлять ли импорт torch
        history_file_path (str): путь к файлу истории
        
    Returns:
        Tuple[str, str]: путь к временному файлу и его содержимое
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_file_path = temp_dir / py_file_path

    # Добавляем необходимые импорты
    if add_imports:
        code = "\n\nimport torch\n\n\n" + code

    # Сохраняем во временный файл
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Читаем содержимое для возврата
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # Сохраняем в историю
    save_to_history(history_file_path, code)
    
    return str(temp_file_path), file_content


def save_to_history(py_file_path: str, code: str, add_imports: bool = True) -> str:
    """
    Добавляет код в файл истории
    
    Args:
        py_file_path (str): путь к файлу истории
        code (str): код для добавления
        add_imports (bool): добавлять ли импорт torch
        
    Returns:
        str: путь к файлу истории
    """
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(py_file_path), exist_ok=True)
    
    # Добавляем разделитель и временную метку
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = f"\n\n#{'='*60}\n# Сохранено: {timestamp}\n#{'='*60}\n\n"
    
    with open(py_file_path, 'a', encoding='utf-8') as f:
        f.write(separator)
        f.write(code)
        
    return py_file_path
