from typing import Any, Dict
import streamlit as st
from .base_config import ModelConfig, DevelopmentConfig, DeploymentConfig


class ConfigLoader:
    """Утилиты для загрузки конфигурации из Streamlit secrets"""
    
    @staticmethod
    def from_streamlit_secrets() -> Dict[str, Any]:
        """Загрузить конфигурацию из Streamlit secrets"""
        try:
            secrets_config = {}
            
            # LLM настройки
            if 'HUGGINGFACE_HUB_TOKEN' in st.secrets:
                secrets_config['llm'] = {
                    'HUGGINGFACE_HUB_TOKEN': st.secrets['HUGGINGFACE_HUB_TOKEN']
                }
                
                # Опциональные LLM настройки
                if 'LLM_MODEL_NAME' in st.secrets:
                    secrets_config['llm']['MODEL_NAME'] = st.secrets['LLM_MODEL_NAME']
                if 'LLM_TEMPERATURE' in st.secrets:
                    secrets_config['llm']['TEMPERATURE'] = float(st.secrets['LLM_TEMPERATURE'])
            
            # Supabase настройки
            if 'SUPABASE_URL' in st.secrets and 'SUPABASE_KEY' in st.secrets:
                secrets_config['supabase'] = {
                    'URL': st.secrets['SUPABASE_URL'],
                    'KEY': st.secrets['SUPABASE_KEY']
                }
            
            # App настройки
            app_config = {}
            if 'APP_MODE' in st.secrets:
                app_config['MODE'] = st.secrets['APP_MODE']
            if 'DEBUG' in st.secrets:
                app_config['DEBUG'] = st.secrets['DEBUG'].lower() == 'true'
            
            if app_config:
                secrets_config['app'] = app_config
            
            # Training настройки
            if 'ENABLE_TENSORBOARD' in st.secrets:
                secrets_config['training'] = {
                    'ENABLE_TENSORBOARD_LOGGING': st.secrets['ENABLE_TENSORBOARD'].lower() == 'true'
                }
                
            return secrets_config
            
        except (FileNotFoundError, KeyError):
            # Fallback к пустому конфигу если secrets не доступны
            return {}
    
    @classmethod
    def create_config(cls, config_type: str = "auto") -> ModelConfig:
        """Фабричный метод для создания конфигурации"""
        
        # Определяем режим работы
        if config_type == "development":
            config = DevelopmentConfig()
        elif config_type == "deployment":
            config = DeploymentConfig()
        else:
            # Автоматическое определение режима
            try:
                if st.secrets.get('DEPLOYMENT', '').lower() == 'true':
                    config = DeploymentConfig()
                else:
                    config = DevelopmentConfig()
            except (FileNotFoundError, KeyError):
                # Если нет доступа к secrets, используем development
                config = DevelopmentConfig()
        
        # Загружаем и применяем настройки из secrets
        secrets_config = cls.from_streamlit_secrets()
        cls._apply_secrets_config(config, secrets_config)
        
        # Валидируем конфигурацию
        try:
            config.validate()
        except ValueError as e:
            print(f"⚠️  Config validation warning: {e}")
        
        return config
    
    @classmethod
    def _apply_secrets_config(cls, config: ModelConfig, secrets_config: Dict[str, Any]):
        """Применить настройки из Streamlit secrets к конфигу"""
        if not secrets_config:
            return
            
        # LLM настройки
        if 'llm' in secrets_config:
            llm_secrets = secrets_config['llm']
            
            # Обязательные поля
            if 'HUGGINGFACE_HUB_TOKEN' in llm_secrets:
                config.llm.HUGGINGFACE_HUB_TOKEN = llm_secrets['HUGGINGFACE_HUB_TOKEN']
            
            # Опциональные поля
            if 'MODEL_NAME' in llm_secrets:
                config.llm.MODEL_NAME = llm_secrets['MODEL_NAME']
            if 'TEMPERATURE' in llm_secrets:
                config.llm.TEMPERATURE = llm_secrets['TEMPERATURE']
        
        # Supabase настройки
        if 'supabase' in secrets_config:
            supabase_secrets = secrets_config['supabase']
            if 'URL' in supabase_secrets:
                config.supabase.URL = supabase_secrets['URL']
            if 'KEY' in supabase_secrets:
                config.supabase.KEY = supabase_secrets['KEY']
        
        # App настройки
        if 'app' in secrets_config:
            app_secrets = secrets_config['app']
            if 'MODE' in app_secrets:
                config.app.MODE = app_secrets['MODE']
            if 'DEBUG' in app_secrets:
                config.app.DEBUG = app_secrets['DEBUG']
        
        # Training настройки
        if 'training' in secrets_config:
            training_secrets = secrets_config['training']
            if 'ENABLE_TENSORBOARD_LOGGING' in training_secrets:
                config.training.ENABLE_TENSORBOARD_LOGGING = training_secrets['ENABLE_TENSORBOARD_LOGGING']


def get_config() -> ModelConfig:
    """Упрощенная функция для получения конфигурации"""
    return ConfigLoader.create_config()