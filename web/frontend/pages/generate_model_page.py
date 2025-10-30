import streamlit as st
from lib.translator import translate

# Импорт конфигурации и контроллеров
from web.backend.config.config_utils import get_config
from web.backend.controllers import ModelGenerationController, ResultsController
from web.backend.utils import compare_metrics, download_temp_file, show_mode_indicator

# Инициализация конфигурации
config = get_config()

# Инициализация контроллеров
generation_controller = ModelGenerationController(config)
results_controller = ResultsController(config)


def generate_model_page():
    show_mode_indicator()
    
    # Данные из session state
    EXPERT_COMMENT = st.session_state.user_comment
    comment_class = st.session_state.user_comment_class
    comment_subclass = st.session_state.user_comment_subclass
    model_type = st.session_state.model_type

    # Создаем контейнеры для отображения прогресса
    st.header(translate("🚀 Процесс генерации и обучения модели"))
    progress_bar = st.progress(0)
    status_text = st.empty()
    details_container = st.empty()
    error_container = st.empty()

    # Callback функция для обновления прогресса
    def progress_callback(message, progress):
        status_text.text(translate(message))
        progress_bar.progress(progress)

    # Запуск пайплайна генерации модели через контроллер
    pipeline_result = generation_controller.execute_training_pipeline(
        expert_comment=EXPERT_COMMENT,
        comment_class=comment_class,
        comment_subclass=comment_subclass,
        model_type=model_type,
        progress_callback=progress_callback
    )

    # Обработка результата пайплайна
    if not pipeline_result['success']:
        error_container.error(translate(f"❌ {pipeline_result['error']}"))
        progress_bar.progress(100)
        st.error(translate("Процесс остановлен из-за ошибок. Пожалуйста, обновите страницу"))
        return

    # Успешное завершение пайплайна
    st.success(translate("Модель успешно сгенерирована и обучена!"))

    # Детали выполнения в режиме разработчика
    if st.session_state.mode == 'DEV':
        with st.expander(translate("Детали выполнения")):
            st.write(translate("**Информация о процессе:**"))
            st.json({
                "validation_attempts": pipeline_result['error_count'],
                "final_code_length": len(pipeline_result['final_code']),
                "model_saved_to": pipeline_result['model_path']
            })

    # Сравнение моделей через ResultsController
    supabase = st.session_state['supabase']
    comparison_result = results_controller.compare_models(
        supabase_client=supabase,
        old_model_path="",  # путь будет получен из конфига внутри контроллера
        new_model_path=pipeline_result['model_path'],
        model_type=model_type
    )

    if not comparison_result['success']:
        st.error(translate(f"Ошибка при сравнении моделей: {comparison_result['error']}"))
        # Показываем кнопки даже если сравнение не удалось
        _show_action_buttons(pipeline_result['model_path'], pipeline_result['loss_path'])
        return

    # Успешное сравнение моделей - показываем результаты
    _show_comparison_results(comparison_result)
    _show_action_buttons(pipeline_result['model_path'], pipeline_result['loss_path'])

    # Стили
    st.markdown(translate("""
        <style>
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlock"]) {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #f0f6c1;
            margin-bottom: 20px;
        }
        </style>
        """), unsafe_allow_html=True)


def _show_comparison_results(comparison_result: dict):
    """Показать результаты сравнения моделей"""
    st.title(translate("Сравнение моделей"))
    
    metrics_old = comparison_result['metrics_old']
    metrics_new = comparison_result['metrics_new']
    figures = comparison_result['figures']
    epidemic_params = comparison_result['epidemic_params']

    # Отображение графиков и метрик
    col1, col2 = st.columns(2)
    OLD_MODEL_NAME = translate("PINN")
    NEW_MODEL_NAME = translate("NEW_PINN")
    
    with col1:
        with st.expander(translate("📈 Susceptible (S) - Развернуть/Свернуть"), expanded=True):
            st.plotly_chart(figures['S'])
            st.write(translate("📊 Метрики моделей для S"))
            comparison_table = compare_metrics(metrics_old['S'], metrics_new['S'], OLD_MODEL_NAME, NEW_MODEL_NAME)

    with col2:
        with st.expander(translate("🦠 Infected (I) - Развернуть/Свернуть"), expanded=True):
            st.plotly_chart(figures['I'])
            st.write(translate("📊 Метрики моделей для I"))
            comparison_table = compare_metrics(metrics_old['I'], metrics_new['I'], OLD_MODEL_NAME, NEW_MODEL_NAME)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander(translate("💊 Recovered (R) - Развернуть/Свернуть"), expanded=True):
            st.plotly_chart(figures['R'])
            st.write(translate("📊 Метрики моделей для R"))
            comparison_table = compare_metrics(metrics_old['R'], metrics_new['R'], OLD_MODEL_NAME, NEW_MODEL_NAME)

    with col2:
        with st.expander(translate("⚰️ Dead (D) - Развернуть/Свернуть"), expanded=True):
            st.plotly_chart(figures['D'])
            st.write(translate("📊 Метрики моделей для D"))
            comparison_table = compare_metrics(metrics_old['D'], metrics_new['D'], OLD_MODEL_NAME, NEW_MODEL_NAME)
    
    # Показ эпидемиологических параметров
    st.subheader(translate("Эпидемиологические параметры"))
    st.metric(translate("R0 (базовое репродуктивное число)"), f"{epidemic_params['r0']:.3f}")
        
    st.plotly_chart(figures['epid'], width='stretch')

    # Рекомендации
    with st.expander(translate("💡 Что делать после получения прогноза?"), expanded=True):
        st.success(translate("✅ **Если прогноз устраивает:**"))
        st.markdown(translate("""
        1. Нажмите кнопку **«Сохранить модель в Storage»**
        2. Вы автоматически перейдете на главную страницу
        3. В боковой панели выберите **«Кастомизированная модель»** для работы с сохраненной моделью
        """))
        
        st.warning(translate("❌ **Если прогноз не устраивает:**"))
        st.markdown(translate("""
        1. Вернитесь на главную страницу
        2. Переформулируйте указания для модели
        """))


def _show_action_buttons(model_path: str, loss_path: str):
    """Показать кнопки действий"""
    # Сохраняем пути в session state
    st.session_state.current_model_path = model_path
    st.session_state.current_loss_path = loss_path
    
    # Кнопки возврата и сохранения
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(translate("↩️ Вернуться на главную страницу"), 
             use_container_width=True,
             on_click=lambda: setattr(st.session_state, 'current_page', 'main')):
            pass
    
    with col2:
        st.button(
            translate("💾 Сохранить модель в Storage"),
            use_container_width=True,
            on_click=save_model_callback,
            key="save_model_btn"
        )

    if 'save_status' in st.session_state:
        st.write(st.session_state['save_status'])
            
    # Скачивание файлов в режиме разработчика
    if st.session_state.mode == 'DEV':
        download_temp_file(loss_path)
        download_temp_file(model_path)


def save_model_callback():
    """Callback для сохранения модели"""
    try:
        supabase = st.session_state['supabase']
        filename = st.session_state.current_model_path
        loss_file_path = st.session_state.get('current_loss_path')

        with open(filename, "rb") as model_file:
            response = (
                supabase.storage
                .from_(config.supabase.STORAGE_BUCKET)
                .upload(
                    file=model_file,
                    path=config.paths.CUSTOM_MODEL_PATH,
                    file_options={"cache-control": "3600", "upsert": "true"}
                )
            )

        with open(loss_file_path, "rb") as loss_file:
            response = (
                supabase.storage
                .from_(config.supabase.STORAGE_BUCKET)
                .upload(
                    file=loss_file,
                    path=config.paths.CUSTOM_LOSS_PATH,
                    file_options={"cache-control": "3600", "upsert": "true"}
                )
            )

        st.success(translate("✅ Модель успешно сохранена!"))
        st.session_state.current_page = "main"
        
    except Exception as e:
        st.error(translate(f"❌ Ошибка при сохранении модели: {str(e)}"))