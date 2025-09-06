from loss_dinn_check import TestLossDinnFunction
import tempfile
import os

def save(py_file_path, code, add_imports=True, history_file_path = 'losses.py'):

    code = "\n\nimport torch\n\n\n" + code  # костыль
    with open(py_file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # нужна проверка на корректность лосса!
    save_to_history(history_file_path, code)
    return py_file_path


def save_py(py_file_path, code, add_imports=True, history_file_path='losses.py'):
    # Создаем временный файл
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, py_file_path)

    code = "\n\nimport torch\n\n\n" + code
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Для скачивания файла пользователем
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    return temp_file_path, file_content

def save_to_history(py_file_path, code, add_imports=True):
    with open(py_file_path, 'a', encoding='utf-8') as f:
        f.write(code)
        
    return py_file_path

