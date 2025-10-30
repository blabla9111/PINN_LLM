import re
import json
from pathlib import Path
from typing import Union, Dict


def llm_answer_get_comment_class(answer: Union[str, Dict]) -> str:
    """
    Извлекает содержимое комментария из ответа LLM
    
    Args:
        answer (Union[str, Dict]): Ответ от LLM в формате JSON или словарь
        
    Returns:
        str: Текст комментария
    """
    if isinstance(answer, str):
        answer_dict = json.loads(answer)
    else:
        answer_dict = answer
        
    return answer_dict['choices'][0]['message']['content']


def llm_answer_to_python_code(answer: Union[str, Dict]) -> str:
    """
    Преобразует ответ LLM в чистый Python код
    
    Args:
        answer (Union[str, Dict]): Ответ от LLM в формате JSON или словарь
        
    Returns:
        str: Извлеченный Python код
    """
    print(f"🔧 Обработка ответа LLM: {answer[:100]}...")
    
    if isinstance(answer, str):
        code = extract_python_code(answer)
    else:
        code = extract_python_code(answer['choices'][0]['message']['content'])
    
    print(f"📝 Извлеченный код ({len(code)} символов)")
    return code


def extract_python_code(llm_response: str) -> str:
    """
    Извлекает Python-код из ответа LLM, обрабатывая оба варианта:
    1. Код в Markdown-блоках ```python ... ```
    2. "Чистый" код без обёрток
    
    Args:
        llm_response (str): Ответ от языковой модели
        
    Returns:
        str: Очищенный Python код
    """
    if not llm_response:
        return ""
        
    # Паттерн для нахождения Markdown-блоков с Python-кодом
    pattern = r'```(?:python)?\n?(.*?)```'
    matches = re.findall(pattern, llm_response, re.DOTALL)

    if matches:
        # Если найден Markdown-блок - возвращаем первый (часто он один)
        extracted_code = matches[0].strip()
        print(f"🎯 Извлечен код из Markdown блока ({len(extracted_code)} символов)")
        return extracted_code
    else:
        # Если блоков нет - проверяем, есть ли вообще код в ответе
        python_keywords = ['def ', 'class ', 'import ', 'return ', 'def ']
        if any(keyword in llm_response for keyword in python_keywords):
            print("📄 Используется чистый код из ответа")
            return llm_response.strip()
        
        print("⚠️ В ответе не обнаружен Python код")
        return ""


def load_text_to_json(file_path: Union[str, Path], to_json: bool = True) -> Union[Dict, str]:
    """
    Загружает текстовый файл и преобразует его в JSON
    
    Args:
        file_path (Union[str, Path]): Путь к файлу
        to_json (bool): Преобразовать ли в JSON объект
        
    Returns:
        Union[Dict, str]: JSON объект или исходный текст
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if to_json:
        return json.loads(content)
    return content


def get_loss_func_as_str(file_path: Union[str, Path]) -> str:
    """
    Читает файл с функцией потерь и возвращает как строку без импортов torch
    
    Args:
        file_path (Union[str, Path]): Путь к файлу с функцией потерь
        
    Returns:
        str: Очищенный код функции потерь
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл с функцией потерь не найден: {file_path}")
    
    # Читаем все строки файла
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Фильтруем строки с импортами torch
    filtered_lines = []
    torch_imports_removed = 0
    
    for line in lines:
        stripped = line.strip()
        # Пропускаем строки, начинающиеся с import torch или from torch
        if stripped.startswith(('import torch', 'from torch')):
            torch_imports_removed += 1
            continue
        filtered_lines.append(line)

    print(f"🧹 Удалено {torch_imports_removed} строк с импортами torch")
    
    return ''.join(filtered_lines)