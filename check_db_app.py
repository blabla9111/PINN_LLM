import streamlit as st
import os
from supabase import create_client, Client
from lib.create_file import *

st.title("🔌 Проверка подключения к Supabase")

try:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        st.error("❌ Не найдены SUPABASE_URL или SUPABASE_KEY")
    else:
        # Просто создаем клиента - это уже проверит подключение
        supabase = create_client(url, key)
        st.success("✅ Подключение к Supabase успешно!")
        test_response = supabase.table("PINN_LLM_MODELS").select("*").execute()
        st.json(test_response.__dict__)
        response = supabase.storage.from_("PINN_LLM_STORAGE").download("dinn_cuda.pth")
        # print(response['data'])
        filename = load_model_to_tmp(response)
        with open(filename, "rb") as model_file:
                    # supabase = create_client(url, key)
                    response = (
                            supabase.storage
                            .from_("PINN_LLM_STORAGE")
                            .upload(
                                file=model_file,
                                path="NEW_MODEL_dinn_cuda_2.pth",
                                file_options={"cache-control": "3600", "upsert": "false"}
                            )
                        )
        st.success("✅ Модель успешно сохранена в Storage!")

        # 'response' will contain the file content as bytes
        # You can then save it to a local file or process it directly
        
        # Example: Save to a local file
        # with open("downloaded_file.txt", "wb") as f:
        #     f.write(response)
        # print(f"File '{file_path_in_storage}' downloaded successfully.")
        
        st.write(f"**URL:** {url}")
        st.write(f"**Key:** {key[:15]}...")
        
except Exception as e:
    st.error(f"❌ Ошибка: {e}")