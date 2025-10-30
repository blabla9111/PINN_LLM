import subprocess
import sys
from lib.create_file import create_file_in_tmp


class ModelTrainingService:
    """Сервис для обучения моделей"""
    
    def __init__(self, config=None):
        from web.backend.config.config_utils import get_config
        self.config = config or get_config()
    
    def train_pinn_model(self, loss_code: str) -> str:
        """Запуск обучения PINN модели"""
        file_path, content = create_file_in_tmp(
            loss_code,
            self.config.paths.PINN_NEW_FILE_NAME,
            self.config.paths.PINN_RUN_START_FILE_PATH,
            self.config.paths.PINN_RUN_END_FILE_PATH
        )
        
        output = subprocess.run(
            [f"{sys.executable}", file_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Извлекаем путь к сохраненной модели
        text = str(output.stdout)
        lines = text.strip().split('\n')
        last_line = lines[-1] if lines else ""
        filename = last_line.split("Model saved to ")[-1].strip()
        
        return filename
    
    def get_training_progress(self) -> dict:
        """Получение прогресса обучения (заглушка для будущей реализации)"""
        return {
            "status": "training",
            "epoch": 0,
            "total_epochs": self.config.training.EPOCHS,
            "loss": 0.0
        }