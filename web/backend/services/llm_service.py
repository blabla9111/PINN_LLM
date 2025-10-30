from web.backend.config.config_utils import get_config
from lib.prompt_sender import send_prompt
from lib.parser import load_text_to_json, llm_answer_to_python_code


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
        from lib.prompt_sender import create_prompt_to_fix_error
        
        create_prompt_to_fix_error(prompt_path, code, error)
        return self.generate_code_from_prompt(prompt_path, answer_path)