import streamlit as st
from deep_translator import GoogleTranslator

class Translator:
    """Умный переводчик с использованием GoogleTranslator"""
    
    def __init__(self):
        self.cache = {}  # Кэш переводов для производительности
        
    def __call__(self, text: str) -> str:
        """
        Основной метод перевода
        
        Args:
            text (str): Текст для перевода
            
        Returns:
            str: Переведенный текст или оригинал, если язык русский
        """
        lang = st.session_state.get('language', 'ru')
        
        # Если язык русский - возвращаем оригинал
        if lang == 'ru':
            return text
            
        # Если язык английский - переводим
        elif lang == 'en':
            return self._translate_to_english(text)
            
        else:
            return text
    
    def _translate_to_english(self, text: str) -> str:
        """
        Переводит текст на английский используя GoogleTranslator
        
        Args:
            text (str): Текст для перевода
            
        Returns:
            str: Переведенный текст
        """
        # Проверяем кэш
        if text in self.cache:
            return self.cache[text]
        
        try:
            # Используем GoogleTranslator
            translated_text = GoogleTranslator(source="ru", target="en").translate(text)
            
            # Сохраняем в кэш
            self.cache[text] = translated_text
            
            return translated_text
            
        except Exception as e:
            st.error(f"Ошибка при переводе: {e}")
            return text
    
    def get_language(self) -> str:
        """
        Получить текущий язык
        
        Returns:
            str: Код языка ('ru' или 'en')
        """
        return st.session_state.get('language', 'en')
    
    def set_language(self, lang: str) -> None:
        """
        Установить язык
        
        Args:
            lang (str): Код языка ('ru' или 'en')
        """
        if lang not in ['ru', 'en']:
            raise ValueError("Language must be 'ru' or 'en'")
        st.session_state.language = lang
    
    def clear_cache(self) -> None:
        """Очистить кэш переводов"""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """Получить размер кэша"""
        return len(self.cache)
    
translate = Translator()