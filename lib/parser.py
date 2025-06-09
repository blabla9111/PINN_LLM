import re
import json

def llm_answer_get_comment_class(answer):
    answer_dict = json.loads(answer)
    return answer_dict['choices'][0]['message']['content']

def llm_answer_to_python_code(answer):
    # answer_json = json.load(answer)
    answer_dict = json.loads(answer)
    code = answer_dict['choices'][0]['message']['content']
    code = extract_python_code(code)
    # if code[0].str().contain("'''python"):
    #     code = code[1:-1]
    # code = '\n'.join(code)
    # print(code)
    return code


def extract_python_code(llm_response: str) -> str:
    """
    Извлекает Python-код из ответа LLM, обрабатывая оба варианта:
    1. Код в Markdown-блоках ```python ... ```
    2. "Чистый" код без обёрток
    """
    # Паттерн для нахождения Markdown-блоков с Python-кодом
    pattern = r'```(?:python)?\n?(.*?)```'
    matches = re.findall(pattern, llm_response, re.DOTALL)

    if matches:
        # Если найден Markdown-блок - возвращаем первый (часто он один)
        return matches[0].strip()
    else:
        # Если блоков нет - проверяем, есть ли вообще код в ответе
        if any(keyword in llm_response.lower() for keyword in ['def ', 'class ', 'import ', 'print ']):
            return llm_response.strip()
        return ""  # Или можно возбудить исключение
    
def load_text_to_json(file_path, to_json = True):
    with open(file_path, 'r') as f:
        json_text = json.load(f)

    return json_text


def get_loss_func_as_str(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Удаляем строки, начинающиеся с import torch или from torch
        if stripped.startswith('import torch') or stripped.startswith('from torch'):
            continue
        filtered_lines.append(line)

    return ''.join(filtered_lines)
