import os
from typing import Dict, Any
from web.backend.utils.prompt_sender_utils import create_get_loss_based_on_recommendation_prompt


class PromptService:
    """Сервис для работы с промптами"""
    
    def __init__(self, config=None):
        from web.backend.config.config_utils import get_config
        self.config = config or get_config()
    
    def create_loss_generation_prompt(self, comment_class: str, comment_subclass: str, 
                                    expert_comment: str, current_code: str) -> str:
        """Создание промпта для генерации функции потерь"""
        create_get_loss_based_on_recommendation_prompt(
            self.config.paths.PROMPT_FILE_PATH,
            comment_class,
            comment_subclass,
            expert_comment,
            current_code
        )
        return self.config.paths.PROMPT_FILE_PATH
    
    def load_current_loss_function(self, model_type: str) -> str:
        """Загрузка текущей функции потерь"""
        from web.backend.utils.parser_utils import get_loss_func_as_str
        from web.backend.utils.create_file_utils import load_python_file_to_tmp
        
        supabase = self._get_supabase()
        loss_storage_path = self.config.get_loss_storage_path(model_type)
        response = supabase.storage.from_(self.config.supabase.STORAGE_BUCKET).download(loss_storage_path)
        
        loss_filepath = load_python_file_to_tmp(response)
        return get_loss_func_as_str(loss_filepath)
    
    def _get_supabase(self):
        """Получение Supabase клиента"""
        import streamlit as st
        return st.session_state['supabase']