from typing import Dict, Any
from web.backend.comment_classificator.match_loss_classification import predict_class_and_sub_class
from web.backend.config.config_utils import get_config


class AnalysisController:
    """Контроллер для анализа экспертных указаний"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
    def analyze_comment(self, comment: str) -> Dict[str, Any]:
        """
        Проанализировать комментарий и определить класс/подкласс
        
        Returns:
            Dict с результатами анализа
        """
        try:
            top_indices, top_probs, is_valid = predict_class_and_sub_class(comment)
            comment_class = str(top_indices[0])
            comment_subclass = str(top_indices[1])
            
            return {
                'success': True,
                'comment_class': comment_class,
                'comment_subclass': comment_subclass,
                'probabilities': {
                    'class': float(top_probs[0]),
                    'subclass': float(top_probs[1])
                },
                'is_valid': is_valid,
                'confidence_class': round(top_probs[0] * 100),
                'confidence_subclass': round(top_probs[1] * 100)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Ошибка при анализе комментария: {str(e)}"
            }
    
    def save_expert_comment(self, supabase_client, comment: str, class_num: str, 
                          subclass_num: str, approved: bool = False) -> Dict[str, Any]:
        """
        Сохранить экспертный комментарий в базу данных
        
        Args:
            supabase_client: клиент Supabase
            comment: комментарий
            class_num: номер класса
            subclass_num: номер подкласса
            approved: подтвержден ли комментарий
            
        Returns:
            Dict с результатом сохранения
        """
        try:
            data = {
                "comment": comment,
                "class": class_num,
                "subclass": subclass_num,
                "approved": approved
            }
            
            response = supabase_client.table("expert_comment").insert(data).execute()
            
            return {
                'success': True,
                'comment_id': response.data[0]['id'] if response.data else None,
                'response': response
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Ошибка при сохранении комментария: {str(e)}"
            }
    
    def update_comment_approval(self, supabase_client, comment_id: int, approved: bool) -> Dict[str, Any]:
        """
        Обновить статус подтверждения комментария
        
        Args:
            supabase_client: клиент Supabase
            comment_id: ID комментария
            approved: новый статус подтверждения
            
        Returns:
            Dict с результатом обновления
        """
        try:
            response = supabase_client.table("expert_comment").update(
                {"approved": approved}
            ).eq("id", comment_id).execute()
            
            return {
                'success': True,
                'response': response
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Ошибка при обновлении комментария: {str(e)}"
            }