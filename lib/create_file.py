import tempfile
import os


def create_file_in_tmp(loss_function_str, py_file_path, start_file_path, end_file_path):
    """
    Создает Python файл с функцией потерь для PINN, используя начало и конец из файлов
    
    Args:
        loss_function_str (str): код функции потерь
        py_file_path (str): имя создаваемого файла
        start_file_path (str): путь к файлу с начальным кодом
        end_file_path (str): путь к файлу с завершающим кодом
    """
    # Создаем временный файл
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, py_file_path)

    try:
        # Читаем начало из файла
        with open(start_file_path, 'r', encoding='utf-8') as f:
            code_start_from_file = f.read()

        # Читаем конец из файла
        with open(end_file_path, 'r', encoding='utf-8') as f:
            code_end_from_file = f.read()

        # Собираем полный код
        full_code = code_start_from_file + "\n\n" + \
            loss_function_str + "\n\n" + code_end_from_file

        # Создаем и заполняем файл
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        # Читаем обратно для скачивания
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        return temp_file_path, file_content

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл не найден: {e}")
    except Exception as e:
        raise Exception(f"Ошибка при создании файла: {e}")


def load_model_to_tmp(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        tmp_file.write(model)
        tmp_path = tmp_file.name

    print("Файл сохранен во временное место:", tmp_path)
    return tmp_path

def load_data_to_tmp(data_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(data_file)
        tmp_path = tmp_file.name

    print("Файл сохранен во временное место:", tmp_path)
    return tmp_path