import streamlit as st
from huggingface_hub import InferenceClient

# Настройка страницы
st.set_page_config(page_title="AI Chat", page_icon="🤖")

# Инициализация клиента в session_state
if 'client' not in st.session_state:
    st.session_state.client = InferenceClient(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        token=st.secrets['HUGGINGFACE_HUB_TOKEN']
    )

# Инициализация истории сообщений
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Заголовок чата
st.title("💬 AI Chat")
st.markdown("Чат с Meta-Llama-3.1-8B-Instruct")

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода сообщения
if prompt := st.chat_input("Введите ваше сообщение..."):
    # Добавление сообщения пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Отображение сообщения пользователя
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Генерация ответа AI
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                # Создание completion
                completion = st.session_state.client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    messages=st.session_state.messages,
                    max_tokens=500,
                    temperature=0.7,
                    stream=True  # Включение потоковой передачи для лучшего UX
                )
                
                # Сбор ответа по частям (для потокового отображения)
                response = st.write_stream(
                    chunk.choices[0].delta.content 
                    for chunk in completion 
                    if chunk.choices[0].delta.content is not None
                )
                
                # Добавление ответа в историю
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")

# Кнопка для очистки истории
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()