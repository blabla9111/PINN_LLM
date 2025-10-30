import streamlit as st
import pandas as pd
from web.backend.controllers import TrainingController

def start_page():
    # Получаем конфиг и клиент из session state
    config = st.session_state['app_config']
    supabase = st.session_state['supabase']
    
    # Инициализация контроллера
    training_controller = TrainingController(config)
    
    # Элементы в боковой панели
    st.sidebar.title("Настройки")
    selected_model = st.sidebar.selectbox("Выберите модель", ["LTS Модель (по умолчанию)", "Кастомизированная модель"])

    # Определяем тип модели
    if selected_model == "Кастомизированная модель":
        model_type = "CUSTOM"
        st.session_state.model_type = "CUSTOM"
    else:
        model_type = "LTS"
        st.session_state.model_type = "LTS"

    selected_mode = st.sidebar.selectbox("Выберите режим работы", ["Пользователь (по умолчанию)", "Разработчик"])

    if selected_mode == "Разработчик":
        st.session_state.mode = "DEV"
    else:
        st.session_state.mode = "USER"

    show_mode_indicator()

    # Загрузка модели и получение прогнозов через контроллер
    result = training_controller.load_model_and_predict(supabase, model_type)
    
    if not result['success']:
        st.error(f"Ошибка при загрузке модели: {result['error']}")
        return

    # Извлекаем результаты из ответа контроллера
    figures = result['figures']
    metrics_df = result['metrics_df']
    r0_value = result['r0_value']
    predictions = result['predictions']
    data_info = result['data_info']

    st.title("Информация о модели и прогноз")

    # Разделение на две колонки
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈Графики")
        # Отображение основного графика
        st.plotly_chart(figures['main'])

    with col2:
        st.header("📊 Метрики моделей")
        
        # Отображение метрик
        if not metrics_df.empty:
            st.dataframe(metrics_df, hide_index=True, width='stretch')
        else:
            st.warning("Не удалось рассчитать метрики")

        st.subheader("Эпид.параметры")
        
        # Расчет эпидемиологических параметров через сервис
        epidemic_params = training_controller.metrics_service.calculate_epidemic_params(
            predictions['S'], predictions['I'], predictions['R'], predictions['D'], 
            range(len(predictions['S']))  # временные метки
        )
        
        st.metric("R0 (basic reproduction number)", f"{r0_value:.3f}")
        
        # Дополнительные метрики
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Пик инфицированных", f"{epidemic_params['peak_infected']:,.0f}")
        with col_metric2:
            st.metric("Общее количество случаев", f"{epidemic_params['total_cases']:,.0f}")

        st.plotly_chart(figures['epid'], width='stretch')

    # Разделитель
    st.divider()

    # Поле для ввода комментария и кнопка отправки
    st.header("Указания для модели")

    comment = st.text_area(
        "Здесь Вы можете написать свой комментарий, если в чем-то не согласны с прогнозом. Желательно выделить главный недостаток прогноза и описать его.",
        placeholder="Введите ваши наблюдения или предложения по прогнозу модели...",
        height=100,
        key="comment_input"
    )

    col_btn1, col_btn2 = st.columns([1, 6])

    with col_btn1:
        if st.button("Отправить", type="primary", key="submit_btn"):
            if comment.strip():
                # Обработка комментария через контроллер
                process_result = training_controller.process_user_comment(comment, st.session_state)
                
                if process_result['success']:
                    # Переключаемся на страницу результатов
                    st.session_state.current_page = "results"
                    st.rerun()
                else:
                    st.error(f"Ошибка при обработке комментария: {process_result['error']}")
            else:
                st.warning("Пожалуйста, введите рекомендации по прогнозу модели перед отправкой")

    # Дополнительная секция с историей комментариев
    with st.expander("История экспертных указаний"):
        if 'comment_history' not in st.session_state:
            st.session_state.comment_history = []

        if st.session_state.comment_history:
            for i, item in enumerate(st.session_state.comment_history[::-1]):
                if st.button(f"Просмотреть подробнее #{len(st.session_state.comment_history)-i}", key=f"view_{i}"):
                    st.session_state.current_page = "results"
                    st.session_state.user_comment = item['comment']
                    st.session_state.comment_analysis = {
                        "main_class": item['main_class'],
                        "subclass": item['subclass']
                    }
                    st.rerun()
        else:
            st.info("История экспертных указаний пуста")

    # Информация о конфигурации в режиме разработчика
    if st.session_state.mode == 'DEV':
        with st.expander("🔧 Информация о конфигурации"):
            st.write("**Текущая конфигурация:**")
            st.json({
                "model": {
                    "type": st.session_state.model_type,
                    "path": training_controller.get_model_storage_path(model_type),
                    "train_size": data_info['train_size']
                },
                "supabase": {
                    "bucket": config.supabase.STORAGE_BUCKET,
                    "url_configured": bool(config.supabase.URL)
                },
                "llm": {
                    "model": config.llm.MODEL_NAME,
                    "temperature": config.llm.TEMPERATURE
                },
                "app": {
                    "mode": config.app.MODE,
                    "debug": config.app.DEBUG,
                    "tensorboard_logging": config.training.ENABLE_TENSORBOARD_LOGGING
                },
                "training": {
                    "max_error_iterations": config.training.MAX_ERROR_ITERATIONS,
                    "epochs": config.training.EPOCHS
                }
            })
            
        # Информация о данных
        with st.expander("📊 Информация о данных"):
            st.write(f"**Размер данных:** {data_info['timesteps_count']} временных точек")
            st.write(f"**Размер обучающей выборки:** {data_info['train_size']}")

# Вспомогательная функция для отображения индикатора режима
def show_mode_indicator():
    if st.session_state.mode == 'DEV':
        st.sidebar.info("🔧 Режим разработчика")
    else:
        st.sidebar.info("Пользовательский режим")