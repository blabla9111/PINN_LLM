from typing import Dict, Any
from web.backend.config.config_utils import get_config
from web.backend.services import MetricsService
from web.backend.utils import (
    load_data_to_tmp, load_model_to_tmp, get_data_for_model,
    plot_S_comparison, plot_I_comparison, plot_R_comparison, plot_D_comparison,
    display_compared_epid_params, load_model
)


class ResultsController:
    """Контроллер для отображения результатов сравнения моделей"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.metrics_service = MetricsService(self.config)
        
    def compare_models(self, 
                      supabase_client,
                      old_model_path: str, 
                      new_model_path: str,
                      model_type: str) -> Dict[str, Any]:
        """
        Сравнить две модели и вернуть результаты
        
        Args:
            supabase_client: клиент Supabase
            old_model_path: путь к старой модели
            new_model_path: путь к новой модели
            model_type: тип модели
            
        Returns:
            Dict с результатами сравнения
        """
        try:
            # Загрузка данных
            response = supabase_client.storage.from_(self.config.supabase.STORAGE_BUCKET).download("data.csv")
            filepath = load_data_to_tmp(response)
            timesteps, susceptible, infected, dead, recovered, x = get_data_for_model(filepath)
            
            # Загрузка моделей
            model_storage_path = self.config.get_model_storage_path(model_type)
            response = supabase_client.storage.from_(self.config.supabase.STORAGE_BUCKET).download(model_storage_path)
            old_model_filepath = load_model_to_tmp(response)
            
            train_size = self.config.training.TRAIN_SIZE
            loaded_dinn_old = load_model(old_model_filepath, timesteps, susceptible, infected, dead, recovered, train_size)
            loaded_dinn_new = load_model(new_model_path, timesteps, susceptible, infected, dead, recovered, train_size)
            
            # Получение предсказаний
            S_pred_old, I_pred_old, D_pred_old, R_pred_old, alpha_pred_old = loaded_dinn_old.predict()
            S_pred_new, I_pred_new, D_pred_new, R_pred_new, alpha_pred_new = loaded_dinn_new.predict()
            
            # Расчет метрик
            true_data = {
                'S': susceptible[x:x+30],
                'I': infected[x:x+30], 
                'R': recovered[x:x+30],
                'D': dead[x:x+30]
            }
            
            pred_data_old = {
                'S': S_pred_old[x:x+30],
                'I': I_pred_old[x:x+30],
                'R': R_pred_old[x:x+30],
                'D': D_pred_old[x:x+30]
            }
            
            pred_data_new = {
                'S': S_pred_new[x:x+30],
                'I': I_pred_new[x:x+30],
                'R': R_pred_new[x:x+30],
                'D': D_pred_new[x:x+30]
            }
            
            metrics_old = self.metrics_service.calculate_all_metrics(true_data, pred_data_old)
            metrics_new = self.metrics_service.calculate_all_metrics(true_data, pred_data_new)
            
            # Создание графиков
            fig_s = plot_S_comparison(timesteps, x, susceptible, S_pred_old, S_pred_new)
            fig_i = plot_I_comparison(timesteps, x, infected, I_pred_old, I_pred_new)
            fig_r = plot_R_comparison(timesteps, x, recovered, R_pred_old, R_pred_new)
            fig_d = plot_D_comparison(timesteps, x, dead, D_pred_old, D_pred_new)
            
            # Расчет эпидемиологических параметров
            epidemic_params = self.metrics_service.calculate_epidemic_params(
                S_pred_new, I_pred_new, R_pred_new, D_pred_new, timesteps
            )
            
            # График сравнения эпид параметров
            epid_fig = display_compared_epid_params(
                S_pred_old, I_pred_old, R_pred_old, D_pred_old, timesteps,
                S_pred_new, I_pred_new, R_pred_new, D_pred_new, timesteps
            )
            
            return {
                'success': True,
                'metrics_old': metrics_old,
                'metrics_new': metrics_new,
                'figures': {
                    'S': fig_s,
                    'I': fig_i, 
                    'R': fig_r,
                    'D': fig_d,
                    'epid': epid_fig
                },
                'epidemic_params': epidemic_params,
                'models_loaded': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Ошибка при сравнении моделей: {str(e)}",
                'models_loaded': False
            }