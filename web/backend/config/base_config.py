import os
from dataclasses import dataclass
from typing import Dict, Optional
from huggingface_hub import InferenceClient

@dataclass
class LLMConfig:
    """Конфигурация для Language Model"""
    MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    TEMPERATURE: float = 1.0
    MAX_TOKENS: int = 2048
    # TIMEOUT: int = 30
   
    # HuggingFace
    HUGGINGFACE_HUB_TOKEN: str = None
    
    def __post_init__(self):
        if self.HUGGINGFACE_HUB_TOKEN is None:
            self.HUGGINGFACE_HUB_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN', '')

@dataclass
class TrainingConfig:
    """Конфигурация для обучения моделей"""
    # Настройки обучения PINN
    TRAIN_SIZE: int = 180
    EPOCHS: int = 10000
    LEARNING_RATE: float = 0.001
    
    # Валидация кода
    MAX_ERROR_ITERATIONS: int = 6
    VALIDATION_TIMEOUT: int = 30
    
    # TensorBoard
    ENABLE_TENSORBOARD_LOGGING: bool = True
    TENSORBOARD_FLUSH_SECS: int = 10


@dataclass
class FilePathsConfig:
    """Конфигурация путей к файлам"""
    # Основные файлы
    LOSS_FILE_PATH: str = 'loss_dinn_LLM.py'
    LOSS_CHECK_FILE_NAME: str = "loss_dinn_check.py"
    PINN_NEW_FILE_NAME: str = "PINN_NEW_MODEL.py"
    
    # Промпты - исправленные пути на существующие
    PROMPT_FILE_PATH: str = 'promts_templates/get_loss_based_on_recommendation_prompt.json'
    ANSWER_FILE_PATH: str = 'promts_templates/get_loss_based_on_recommendation_prompt_answer.json'
    PROMPT_FIX_ERROR_FILE_PATH: str = "promts_templates/prompt_fix_error.json"
    ANSWER_FIX_ERROR_FILE_PATH: str = 'promts_templates/answer_fix_error_from_LLM_2.json'
    
    # Шаблоны кода - исправленные пути на существующие
    LOSS_CHECK_START_FILE_PATH: str = "web/backend/code_constructor_files/loss_dinn_check/loss_dinn_check_start.txt"
    LOSS_CHECK_END_FILE_PATH: str = "web/backend/code_constructor_files/loss_dinn_check/loss_dinn_check_end.txt"
    PINN_RUN_START_FILE_PATH: str = "web/backend/code_constructor_files/PINN_run/PINN_class_start_code.txt"
    PINN_RUN_END_FILE_PATH: str = "web/backend/code_constructor_files/PINN_run/PINN_class_end_code.txt"
    
    # Модели
    CUSTOM_MODEL_PATH: str = "NEW_MODEL_dinn_cuda_2.pth"
    PRIMARY_MODEL_PATH: str = "dinn_cuda_03_10.pth"
    CUSTOM_LOSS_PATH: str = "loss/loss_dinn_custom.py"
    PRIMARY_LOSS_PATH: str = "loss/loss_dinn_primary.py"

@dataclass
class SupabaseConfig:
    """Конфигурация Supabase"""
    STORAGE_BUCKET: str = "PINN_LLM_STORAGE"
    URL: str = None
    KEY: str = None
    
    def __post_init__(self):
        self.URL = os.getenv('SUPABASE_URL', '')
        self.KEY = os.getenv('SUPABASE_KEY', '')

@dataclass
class AppConfig:
    """Основная конфигурация приложения"""
    # Режимы работы
    MODE: str = "DEV"  # DEV | PROD
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Настройки Streamlit
    PAGE_TITLE: str = "PINN Model Generator"
    PAGE_LAYOUT: str = "wide"
    
    def __post_init__(self):
        # Автоматически определяем режим
        if os.getenv('STREAMLIT_DEPLOYMENT', '').lower() == 'true':
            self.MODE = "PROD"
            self.DEBUG = False

@dataclass
class ModelConfig:
    """Основной класс конфигурации, объединяющий все настройки"""
    def __init__(
        self,
        enable_tensorboard: Optional[bool] = None,
        mode: Optional[str] = None
    ):
        # Инициализация компонентов конфигурации
        self.app = AppConfig()
        self.llm = LLMConfig()
        self.training = TrainingConfig()
        self.paths = FilePathsConfig()
        self.supabase = SupabaseConfig()
        # Переопределение настроек из параметров
        if enable_tensorboard is not None:
            self.training.ENABLE_TENSORBOARD_LOGGING = enable_tensorboard
        if mode is not None:
            self.app.MODE = mode
        # Пост-обработка
        self._setup_logging()

    def get_llm_client(self):
        """Создать LLM клиент с правильными credentials"""
        return InferenceClient(
            model=self.llm.MODEL_NAME,
            token=self.llm.HUGGINGFACE_HUB_TOKEN
        )

    def _setup_logging(self):
        """Настройка логирования на основе конфигурации"""
        import logging
        
        log_level = getattr(logging, self.app.LOG_LEVEL, logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @property
    def progress_steps(self) -> Dict[str, int]:
        """Шаги прогресса для UI"""
        return {
            'prompt_preparation': 10,
            'llm_request': 30,
            'response_processing': 40,
            'code_saving': 50,
            'error_correction': 80,
            'training': 90,
            'complete': 100
        }
    
    @property
    def log_dir(self) -> str:
        """Директория для логов TensorBoard"""
        return self.llm.MODEL_NAME.split("/")[1]
    
    def get_model_storage_path(self, model_type: str) -> str:
        """Получить путь к модели в хранилище"""
        if model_type == "CUSTOM":
            return self.paths.CUSTOM_MODEL_PATH
        else:
            return self.paths.PRIMARY_MODEL_PATH
    
    def get_loss_storage_path(self, model_type: str) -> str:
        """Получить путь к функции потерь в хранилище"""
        if model_type == "CUSTOM":
            return self.paths.CUSTOM_LOSS_PATH
        else:
            return self.paths.PRIMARY_LOSS_PATH
    
    def validate(self) -> bool:
        """Валидация конфигурации"""
        if not self.llm.HUGGINGFACE_HUB_TOKEN:
            raise ValueError("HUGGINGFACE_HUB_TOKEN не установлен")
        if not self.supabase.URL or not self.supabase.KEY:
            raise ValueError("Supabase credentials не установлены")
        return True

class DevelopmentConfig(ModelConfig):
    """Конфигурация для разработки"""   
    def __init__(self):
        super().__init__(
            enable_tensorboard=True,
            mode="DEV"
        )
        # Дополнительные настройки для разработки
        self.training.EPOCHS = 1000  # Меньше эпох для быстрого тестирования
        self.llm.TIMEOUT = 60  # Больше timeout для отладки

class DeploymentConfig(ModelConfig):
    """Конфигурация для деплоя"""
    def __init__(self):
        super().__init__(
            enable_tensorboard=False,  # ❌ Отключаем TensorBoard в продакшене
            mode="PROD"
        )
        # Оптимизации для продакшена
        self.app.DEBUG = False
        self.training.MAX_ERROR_ITERATIONS = 3  # Меньше попыток исправления ошибок