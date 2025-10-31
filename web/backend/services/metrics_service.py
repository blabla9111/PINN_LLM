import pandas as pd
from typing import Dict, List
import streamlit as st
from web.backend.utils import calculate_metrics, get_R0


class MetricsService:
    """Сервис для расчета и отображения метрик"""
    
    def __init__(self, config=None):
        from web.backend.config.config_utils import get_config
        self.config = config or get_config()
    
    def calculate_all_metrics(self, true_data: Dict[str, List], pred_data: Dict[str, List]) -> Dict[str, Dict]:
        """Расчет всех метрик для сравнения моделей"""
        metrics = {}
        
        for key in true_data.keys():
            if key in pred_data:
                metrics[key] = calculate_metrics(true_data[key], pred_data[key])
        
        return metrics
    
    def create_metrics_dataframe(self, metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Создание DataFrame с метриками для отображения"""
        if not metrics_dict:
            return pd.DataFrame()
            
        # Берем ключи из первого элемента
        first_key = next(iter(metrics_dict))
        metric_names = list(metrics_dict[first_key].keys())
        
        data = {'Метрика': metric_names}
        for model_name, metrics in metrics_dict.items():
            data[model_name] = [metrics[metric] for metric in metric_names]
        
        return pd.DataFrame(data)
    
    def calculate_epidemic_params(self, S_pred, I_pred, R_pred, D_pred, timesteps):
        """Расчет эпидемиологических параметров"""
        r0_value = get_R0(S_pred, I_pred, R_pred, D_pred, timesteps).numpy()
        return {
            "r0": float(r0_value),
            "peak_infected": float(I_pred.max()),
            "total_cases": float(I_pred.sum() + R_pred.sum() + D_pred.sum())
        }