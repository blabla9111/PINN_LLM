from web.backend.config.config_utils import get_config
from web.backend.utils.prompt_sender_utils import send_prompt
from web.backend.utils.parser_utils import load_text_to_json, llm_answer_to_python_code
import json


SYSTEM_PROMPT_COMMENT_VALIDATION = """
Ты проверяешь экспертные комментарии к эпидемиологическим прогнозам.

Твоя задача — строго определить, содержит ли комментарий конкретную, полезную и корректную информацию для изменения прогноза.

Критерии правильного комментария:
1. **Конкретика:** комментарий должен описывать чёткое наблюдение или проблему в графике — например, "пик смещён на 10 дней", "слишком резкий спад после пика", "скорость распространения занижена". 
   Недопустимы общие фразы вроде: "прогноз неверный", "надо пересчитать", "график не похож на реальность", "всё неправильно".
2. **Цель:** должна быть явно выражена цель — уточнить, исправить, скорректировать или изменить конкретный аспект прогноза.
3. **Фокус:** в комментарии должна быть указана только одна ключевая проблема.
4. **Тон:** без эмоций, субъективных суждений, выражений недовольства, предположений без аргументов.
5. **Содержательность:** комментарий должен содержать достаточно данных, чтобы можно было изменить прогноз — например, указание на временной диапазон, направление коррекции, темп, процент или числовую оценку. 
   Недопустимы фразы без контекста (“надо скорректировать график”, “что-то не так с пиком”, “эпидемия не так развивается”).
6. **Ясность формулировки:** предложение должно быть грамматически и логически цельным, без двусмысленностей и намёков.

Важно:
- Не добавляй пояснений, текста, вводных слов или комментариев вне JSON.
- Ответ возвращай только в виде корректного JSON-объекта.


Если комментарий не соответствует хотя бы одному из этих пунктов, верни:
{
  "is_valid": false,
  "reason": "укажи, какие пункты нарушены и почему",
  "normalized_comment": "",
  "question": "задай конкретный уточняющий вопрос эксперту, чтобы он мог сделать комментарий корректным"
}

Если комментарий корректен, верни:
{
  "is_valid": true,
  "reason": "",
  "normalized_comment": "отредактированная, чёткая версия комментария"
}
"""

class LLMService:
    """Сервис для работы с Large Language Models"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = self.config.get_llm_client()
    
    def generate_code_from_prompt(self, prompt_path: str, answer_path: str) -> str:
        """Генерация кода на основе промпта"""
        send_prompt(
            prompt_path,
            self.config.llm.MODEL_NAME,
            self.client,
            answer_path,
            temperature=self.config.llm.TEMPERATURE
        )
        
        json_text = load_text_to_json(answer_path)
        return llm_answer_to_python_code(json_text)
    
    def fix_code_error(self, code: str, error: str, prompt_path: str, answer_path: str) -> str:
        """Исправление ошибок в коде"""
        from web.backend.utils.prompt_sender_utils import create_prompt_to_fix_error
        
        create_prompt_to_fix_error(prompt_path, code, error)
        return self.generate_code_from_prompt(prompt_path, answer_path)
    
    def validate_comment(self, history: list, new_comment: str) -> dict:
        """Отправка текущего комментария + истории в LLM"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT_COMMENT_VALIDATION}]
        for role, msg in history:
            messages.append({"role": role, "content": msg})
        messages.append({"role": "user", "content": new_comment})

        output = self.client.chat_completion(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.2
        )
        content = output.choices[0].message["content"]

        try:
            parsed = json.loads(content.strip())
        except json.JSONDecodeError:
            parsed = {
                "is_valid": False,
                "reason": "LLM вернул некорректный JSON:\n" + content,
                "normalized_comment": "",
                "examples": [],
                "question": "Попробуйте уточнить свой комментарий."
            }
        return parsed
