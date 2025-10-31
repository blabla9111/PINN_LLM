import time
from typing import Tuple, Dict, Any
from web.backend.utils.tensorboard_logger import TensorBoardExperimentLogger

from web.backend.config.config_utils import get_config
from web.backend.services import LLMService, CodeValidationService, ModelTrainingService, PromptService


class ModelGenerationController:
    """Контроллер для процесса генерации и обучения моделей"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm_service = LLMService(self.config)
        self.validation_service = CodeValidationService(self.config)
        self.training_service = ModelTrainingService(self.config)
        self.prompt_service = PromptService(self.config)
        
    def execute_training_pipeline(self, 
                                expert_comment: str,
                                comment_class: str, 
                                comment_subclass: str,
                                model_type: str,
                                progress_callback=None) -> Dict[str, Any]:
        """
        Выполнить полный пайплайн генерации и обучения модели
        
        Returns:
            Dict с результатами: {
                'success': bool,
                'model_path': str,
                'loss_path': str, 
                'error_count': int,
                'final_code': str,
                'error': str (если success=False)
            }
        """
        try:
            # Шаг 1: Подготовка промпта
            if progress_callback:
                progress_callback("📝 Подготовка промпта для LLM...", 
                                self.config.progress_steps['prompt_preparation'])
            
            current_code = self.prompt_service.load_current_loss_function(model_type)
            loss_before = current_code
            
            self.prompt_service.create_loss_generation_prompt(
                comment_class, comment_subclass, expert_comment, current_code
            )
            time.sleep(0.5)

            # Шаг 2: Генерация кода
            if progress_callback:
                progress_callback("🤖 Отправка запроса к языковой модели...", 
                                self.config.progress_steps['llm_request'])
            
            generated_code = self.llm_service.generate_code_from_prompt(
                self.config.paths.PROMPT_FILE_PATH,
                self.config.paths.ANSWER_FILE_PATH
            )
            time.sleep(1)

            # Шаг 3: Сохранение кода
            if progress_callback:
                progress_callback("💾 Сохранение сгенерированного кода...", 
                                self.config.progress_steps['code_saving'])
            
            from web.backend.utils.loss_update_utils import save_py
            loss_file_path, content = save_py(self.config.paths.LOSS_FILE_PATH, generated_code)

            # Шаг 4: Валидация с исправлением ошибок
            if progress_callback:
                progress_callback("🧪 Проверка кода...", 
                                self.config.progress_steps['code_saving'] + 10)
            
            final_code, error_count, is_success = self._validate_with_retry(
                generated_code, progress_callback
            )

            if not is_success:
                self._log_tensorboard(expert_comment, loss_before, final_code, error_count, False)
                return {
                    'success': False,
                    'error': f"Не удалось исправить ошибки после {error_count} попыток",
                    'error_count': error_count
                }

            # Шаг 5: Обучение модели
            if progress_callback:
                progress_callback("🏃‍♂️ Запуск обучения PINN модели...", 
                                self.config.progress_steps['training'])
            
            model_path = self.training_service.train_pinn_model(final_code)

            # Логирование успеха
            self._log_tensorboard(expert_comment, loss_before, final_code, error_count, True)

            if progress_callback:
                progress_callback("✅ Обучение завершено!", 
                                self.config.progress_steps['complete'])

            return {
                'success': True,
                'model_path': model_path,
                'loss_path': loss_file_path,
                'error_count': error_count,
                'final_code': final_code
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Критическая ошибка: {str(e)}"
            }
    
    def _validate_with_retry(self, initial_code: str, progress_callback=None) -> Tuple[str, int, bool]:
        """Валидация кода с повторными попытками исправления"""
        error_counter = 0
        current_code = initial_code
        is_correct = False
        error = ""

        # Первоначальная проверка
        is_correct, error = self.validation_service.validate_loss_function(current_code)
        
        # Цикл исправления ошибок
        while not is_correct and error_counter < self.config.training.MAX_ERROR_ITERATIONS:
            error_counter += 1
            
            if progress_callback:
                progress_value = 50 + (error_counter * 10)
                progress_callback(f"⚠️ Исправление ошибок (попытка {error_counter})...", 
                                min(progress_value, self.config.progress_steps['error_correction']))

            # Исправление ошибки
            from web.backend.utils.prompt_sender_utils import create_prompt_to_fix_error
            from web.backend.utils.loss_update_utils import save_py
            
            create_prompt_to_fix_error(
                self.config.paths.PROMPT_FIX_ERROR_FILE_PATH, 
                current_code, 
                error
            )

            if error_counter % 3 == 0:
                # Повторная генерация с основным промптом
                current_code = self.llm_service.generate_code_from_prompt(
                    self.config.paths.PROMPT_FILE_PATH,
                    self.config.paths.ANSWER_FILE_PATH
                )
            else:
                # Исправление конкретной ошибки
                current_code = self.llm_service.generate_code_from_prompt(
                    self.config.paths.PROMPT_FIX_ERROR_FILE_PATH,
                    self.config.paths.ANSWER_FIX_ERROR_FILE_PATH
                )

            # Сохраняем исправленный код
            save_py(self.config.paths.LOSS_FILE_PATH, current_code)

            # Проверка исправленного кода
            is_correct, error = self.validation_service.validate_loss_function(current_code)

        return current_code, error_counter, is_correct
    
    def _log_tensorboard(self, expert_comment: str, loss_before: str, loss_after: str, 
                        error_count: int, is_fixed: bool):
        """Логирование в TensorBoard если включено"""
        if self.config.training.ENABLE_TENSORBOARD_LOGGING:
            experiment_logger = TensorBoardExperimentLogger(log_dir=self.config.log_dir)
            experiment_logger.save_expert_comment(expert_comment)
            experiment_logger.save_loss_func(loss_before, loss_after)
            experiment_logger.save_LLM_model_name(
                self.config.llm.MODEL_NAME, 
                self.config.llm.TEMPERATURE
            )
            experiment_logger.save_loss_function_error_counter(
                error_counter=error_count, 
                is_fixed=is_fixed
            )