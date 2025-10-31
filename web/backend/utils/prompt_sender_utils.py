import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


def send_prompt(prompt_path: Union[str, Path], 
                llm_model_name: str, 
                client: Any, 
                file_path_to_save: Optional[Union[str, Path]] = None, 
                temperature: float = 0) -> Any:
    """
    Отправляет промпт в языковую модель и сохраняет ответ
    
    Args:
        prompt_path: Путь к файлу с промптом
        llm_model_name: Название модели LLM
        client: Клиент для работы с LLM API
        file_path_to_save: Путь для сохранения ответа
        temperature: Температура для генерации
        
    Returns:
        Ответ от языковой модели
    """
    prompt_path = Path(prompt_path)
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Файл промпта не найден: {prompt_path}")
    
    # Загружаем промпт
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)

    print(f"🚀 Отправка запроса к модели {llm_model_name}")
    print(f"💭 Промпт: {prompt_data['messages'][:100]}...")

    # Отправляем запрос к LLM
    completion = client.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": str(prompt_data["messages"])}],
        max_tokens=1024,
        temperature=temperature
    )

    print(f"✅ Получен ответ от LLM")
    print(f"📝 Ответ: {completion.choices[0].message.content[:100]}...")

    # Сохраняем ответ если указан путь
    if file_path_to_save:
        save_answer(file_path_to_save, completion.choices[0].message.content)

    return completion


def save_answer(file_path_to_save: Union[str, Path], text: str) -> str:
    """
    Сохраняет текст ответа в файл
    
    Args:
        file_path_to_save: Путь для сохранения файла
        text: Текст для сохранения
        
    Returns:
        Сохраненный текст
    """
    file_path = Path(file_path_to_save)
    
    # Создаем директорию если не существует
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False, indent=4)
    
    print(f"💾 Ответ сохранен в: {file_path}")
    return text


def create_comment_class_prompt(prompt_file_path: Union[str, Path], comment: str) -> str:
    """
    Создает промпт для классификации комментария по основному классу
    
    Args:
        prompt_file_path: Путь к файлу промпта
        comment: Комментарий пользователя
        
    Returns:
        Путь к обновленному файлу промпта
    """
    prompt_path = Path(prompt_file_path)
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data['messages'][1]['content'] = comment
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"📋 Создан промпт для классификации комментария ({len(comment)} символов)")
    return str(prompt_path)


def create_comment_subclass_prompt(prompt_file_path: Union[str, Path], 
                                 comment_class: str, 
                                 comment: str) -> str:
    """
    Создает промпт для классификации комментария по подклассу
    
    Args:
        prompt_file_path: Путь к файлу промпта
        comment_class: Основной класс комментария
        comment: Комментарий пользователя
        
    Returns:
        Путь к обновленному файлу промпта
    """
    prompt_path = Path(prompt_file_path)
    subclass_rules_path = Path("rules/comment_subclasses.json")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(subclass_rules_path, 'r', encoding='utf-8') as f:
        comment_subclasses = json.load(f)
    
    # Формируем системный промпт
    system_prompt = (
        "You are a helpful assistant with deep knowledge of epidemiology. "
        "Your task is to classify comment. " + 
        comment_subclasses[comment_class] +
        "\nA comment should be assigned to only one class.\n"
        "You get a comment, but you only have to return its class number."
    )
    
    data['messages'][0]['content'] = system_prompt
    data['messages'][1]['content'] = comment
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"📋 Создан промпт для подкласса {comment_class}")
    return str(prompt_path)


def create_primary_prompt(prompt_file_path: Union[str, Path], code: str, comment: str) -> str:
    """
    Создает основной промпт для генерации функции потерь
    
    Args:
        prompt_file_path: Путь к файлу промпта
        code: Исходный код функции потерь
        comment: Комментарий эксперта
        
    Returns:
        Путь к обновленному файлу промпта
    """
    prompt_path = Path(prompt_file_path)
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    content = f"loss function:\n{code}\nExpert comment: \n{comment}\n"
    data['messages'][1]['content'] = content
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"📋 Создан основной промпт ({len(code)} символов кода, {len(comment)} символов комментария)")
    return str(prompt_path)


def create_get_loss_based_on_recommendation_prompt(prompt_file_path: Union[str, Path], 
                                                 comment_class: str, 
                                                 comment_subclass: str, 
                                                 comment: str, 
                                                 code: str) -> None:
    """
    Создает промпт для генерации функции потерь на основе рекомендаций
    
    Args:
        prompt_file_path: Путь к файлу промпта
        comment_class: Основной класс комментария
        comment_subclass: Подкласс комментария
        comment: Комментарий эксперта
        code: Исходный код функции потерь
    """
    prompt_path = Path(prompt_file_path)
    recommendations_path = Path("rules/recommendations_for_class_subclass.json")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(recommendations_path, 'r', encoding='utf-8') as f:
        recommendations = json.load(f)
    
    # Формируем детальный системный промпт
    system_prompt = (
        "You are a machine learning expert specializing in Physics-Informed Neural Networks (PINNs) "
        "and epidemiological modeling. Your task is to modify a loss function for a SIRD model "
        "to better align with epidemiological dynamics.\n"
        "Follow these guidelines strictly:" +
        recommendations[comment_class]["class_info"] +
        recommendations[comment_class][comment_subclass] +
        recommendations[comment_class]["additional_info"] +
        "\n\nTensor specifications:\n"
        "f1: equation for dS/dt, shape: torch.Size([n, m])\n"
        "f2: equation for dI/dt, shape: torch.Size([n, m])\n" 
        "f3: equation for dR/dt, shape: torch.Size([n])\n"
        "f4: equation for dD/dt, shape: torch.Size([n])\n"
        "S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4 are tensors\n"
        "I_pred_last is a 0-d tensor\n"
        "loss should be tensor(float_num, dtype=torch.float64, grad_fn=<AddBackward0>)\n\n"
        "Function signature:\n"
        "loss = loss_dinn(self.S_hat[:self.train_size], S_pred, self.I_hat[:self.train_size], I_pred, "
        "self.D_hat[:self.train_size], D_pred, self.R_hat[:self.train_size], R_pred, "
        "f1[:self.train_size], f2[:self.train_size], f3[:self.train_size], f4[:self.train_size], "
        "I_pred[-1], self.train_size)\n\n"
        "Start your answer with: 'def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, "
        "R_hat, R_pred, f1, f2, f3, f4, I_pred_last, train_size):'\n"
        "Return only the code."
    )
    
    user_content = f"loss function:\n{code}\nExpert comment: \n{comment}"
    
    data['messages'][0]['content'] = system_prompt
    data['messages'][1]['content'] = user_content
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"🎯 Создан промпт с рекомендациями для класса {comment_class}.{comment_subclass}")


def create_prompt_to_fix_error(prompt_file_path: Union[str, Path], code: str, error: str) -> None:
    """
    Создает промпт для исправления ошибок в коде
    
    Args:
        prompt_file_path: Путь к файлу промпта
        code: Код с ошибкой
        error: Текст ошибки
    """
    prompt_path = Path(prompt_file_path)
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    content = f"A previous Python solution code:\n{code}\nProblem trace: {error}.\n** return only code.**"
    data['messages'][1]['content'] = content
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"🔧 Создан промпт для исправления ошибки: {error[:100]}...")