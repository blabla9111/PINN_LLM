import subprocess
import sys
from typing import Tuple
from web.backend.utils.create_file_utils import create_file_in_tmp


class CodeValidationService:
    """Сервис для валидации и проверки кода"""
    
    def __init__(self, config=None):
        from web.backend.config.config_utils import get_config
        self.config = config or get_config()
    
    def validate_loss_function(self, code: str) -> Tuple[bool, str]:
        """Валидация функции потерь"""
        file_path, content = create_file_in_tmp(
            code,
            self.config.paths.LOSS_CHECK_FILE_NAME,
            self.config.paths.LOSS_CHECK_START_FILE_PATH,
            self.config.paths.LOSS_CHECK_END_FILE_PATH
        )
        
        output = subprocess.run(
            [f"{sys.executable}", file_path, code],
            capture_output=True,
            timeout=self.config.training.VALIDATION_TIMEOUT
        )
        
        is_valid = "True" in str(output.stdout)
        error = "" if is_valid else str(output.stdout)
        
        return is_valid, error
    
    def validate_with_retry(self, code: str, max_attempts: int = None) -> Tuple[str, int, bool]:
        """Валидация с повторными попытками исправления"""
        max_attempts = max_attempts or self.config.training.MAX_ERROR_ITERATIONS
        
        error_counter = 0
        current_code = code
        
        while error_counter < max_attempts:
            is_valid, error = self.validate_loss_function(current_code)
            
            if is_valid:
                return current_code, error_counter, True
                
            error_counter += 1
            
        return current_code, error_counter, False