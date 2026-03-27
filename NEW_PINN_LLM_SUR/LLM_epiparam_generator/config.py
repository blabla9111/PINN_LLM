import os
from dotenv import load_dotenv, find_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv(find_dotenv(usecwd=True))

# Берем параметры модели из переменных окружения, если они есть
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME_HF", "meta-llama/Llama-3.3-70B-Instruct")
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE_HF", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
