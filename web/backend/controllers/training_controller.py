from typing import Any, Dict, List, Optional

import pandas as pd

from web.backend.config.config_utils import get_config
from web.backend.services import MetricsService
from web.backend.utils import (
    calculate_metrics,
    display_epid_params,
    get_data_for_model,
    get_R0,
    load_data_to_tmp,
    load_model_to_tmp,
    plot_sidr_predictions_plotly,
    translate_to_en,
    load_model
)


class TrainingController:
    """Контроллер для главной страницы (обучение и прогноз)"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.metrics_service = MetricsService(self.config)
        
    def load_model_and_predict(self, supabase_client, model_type: str) -> Dict[str, Any]:
        """
        Загрузить модель и получить прогнозы
        
        Args:
            supabase_client: клиент Supabase
            model_type: тип модели ("CUSTOM" или "LTS")
            
        Returns:
            Dict с результатами прогноза
        """
        try:
            # Загрузка данных
            response = supabase_client.storage.from_(self.config.supabase.STORAGE_BUCKET).download("data.csv")
            filepath = load_data_to_tmp(response)
            timesteps, susceptible, infected, dead, recovered, x = get_data_for_model(filepath)
            
            # Загрузка модели
            model_storage_path = self.config.get_model_storage_path(model_type)
            response = supabase_client.storage.from_(self.config.supabase.STORAGE_BUCKET).download(model_storage_path)
            filepath = load_model_to_tmp(response)
            
            train_size = self.config.training.TRAIN_SIZE
            loaded_dinn = load_model(filepath, timesteps, susceptible, infected, dead, recovered, train_size)
            S_pred, I_pred, D_pred, R_pred, alpha_pred = loaded_dinn.predict()
            
            # Расчет метрик
            metrics_S = calculate_metrics(susceptible[x:x+30], S_pred[x:x+30])
            metrics_I = calculate_metrics(infected[x:x+30], I_pred[x:x+30])
            metrics_R = calculate_metrics(recovered[x:x+30], R_pred[x:x+30])
            metrics_D = calculate_metrics(dead[x:x+30], D_pred[x:x+30])
            
            # Создание графика
            fig = plot_sidr_predictions_plotly(
                timesteps=timesteps,
                x=x,
                susceptible=susceptible,
                infected=infected,
                dead=dead,
                recovered=recovered,
                S_pred=S_pred,
                I_pred=I_pred,
                D_pred=D_pred,
                R_pred=R_pred,
                start_day='2020-05-07'
            )
            
            # Расчет эпидемиологических параметров
            r0_value = get_R0(S_pred, I_pred, R_pred, D_pred, timesteps).numpy()
            epid_fig = display_epid_params(S_pred, I_pred, R_pred, D_pred, timesteps)
            
            # Создание DataFrame с метриками
            metrics_df = pd.DataFrame({
                'Метрика': list(metrics_I.keys()),
                'S (Susceptible)': list(metrics_S.values()),
                'I (Infected)': list(metrics_I.values()),
                'R (Recovered)': list(metrics_R.values()),
                'D (Dead)': list(metrics_D.values())
            })
            
            return {
                'success': True,
                'figures': {
                    'main': fig,
                    'epid': epid_fig
                },
                'metrics_df': metrics_df,
                'r0_value': float(r0_value),
                'predictions': {
                    'S': S_pred,
                    'I': I_pred,
                    'R': R_pred,
                    'D': D_pred
                },
                'data_info': {
                    'timesteps_count': len(timesteps),
                    'train_size': train_size
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Ошибка при загрузке модели: {str(e)}"
            }
    
    def process_user_comment(self, comment: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработать пользовательский комментарий
        
        Args:
            comment: комментарий пользователя
            session_state: состояние сессии для обновления
            
        Returns:
            Dict с результатами обработки
        """
        try:
            if not comment.strip():
                return {
                    'success': False,
                    'error': "Пустой комментарий"
                }
            
            # Обновляем session state (переданный извне)
            session_state['comment_analysis'] = "analysis_result"
            session_state['comment_primary'] = comment
            session_state['user_comment'] = translate_to_en(comment)
            
            return {
                'success': True,
                'translated_comment': session_state['user_comment']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Ошибка при обработке комментария: {str(e)}"
            }
    
    def get_model_storage_path(self, model_type: str) -> str:
        """Получить путь к модели в хранилище"""
        return self.config.get_model_storage_path(model_type)