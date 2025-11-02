from web.backend.config.config_utils import get_config
from web.backend.services import LLMService

class ValidationController():
    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = self.config.get_llm_client()
        self.llm_service = LLMService(self.config)

    def expert_comment_validation(self, history: list, expert_comment: str):
        return self.llm_service.validate_comment(history, expert_comment)

