import streamlit as st
from web.backend.utils import *
from lib.loss_update import save, save_py
from lib.create_file import create_file_in_tmp
from lib.prompt_sender import *
from lib.parser import *
import subprocess
import time
from huggingface_hub import InferenceClient
import sys
import matplotlib.pyplot as plt
from lib.create_file import *
from lib.TensorBoardExperimentLogger import TensorBoardExperimentLogger

# LLM_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
client = InferenceClient(model=LLM_MODEL_NAME,
                         token=st.secrets['HUGGINGFACE_HUB_TOKEN'])


def generate_model_page():
    show_mode_indicator()
    
    PROMPT_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt.json'
    ANSWER_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt_answer.json'
    LOSS_FILE_PATH = 'loss_dinn_LLM.py'
    # LOSS_PRIMARY_FILE_PATH = 'web/backend/PINN_utils/loss_dinn_primary.py'
    LLM_URL = 'http://localhost:1234/v1/chat/completions'

    LOSS_CHECK_FILE_NAME = "loss_dinn_check.py"
    LOSS_CHECK_START_FILE_PATH = "web/backend/code_constructor_files/loss_dinn_check/loss_dinn_check_start.txt"
    LOSS_CHECK_END_FILE_PATH = "web/backend/code_constructor_files/loss_dinn_check/loss_dinn_check_end.txt"

    PINN_NEW_FILE_NAME = "PINN_NEW_MODEL.py"
    PINN_RUN_START_FILE_PATH = "web/backend/code_constructor_files/PINN_run/PINN_class_start_code.txt"
    PINN_RUN_END_FILE_PATH = "web/backend/code_constructor_files/PINN_run/PINN_class_end_code.txt"


    RUN_PINN_COMMAND = ['python', 'PINN.py']
    RUN_TESTER_COMMAND = ['python', 'loss_dinn_check.py ', '']
    PROMPT_FIX_ERROR_FILE_PATH = "promts_templates/prompt_fix_error.json"
    ANSWER_FIX_ERROR_FILE_PATH = 'promts_templates/answer_fix_error_from_LLM_2.json'

    EXPERT_COMMENT = st.session_state.user_comment
    comment_class = st.session_state.user_comment_class
    comment_subclass = st.session_state.user_comment_subclass

    # Создаем контейнеры для отображения прогресса
    st.header("🚀 Процесс генерации и обучения модели")
    progress_bar = st.progress(0)
    status_text = st.empty()
    details_container = st.empty()
    error_container = st.empty()

    # Шаг 1: Подготовка промпта
    status_text.text("📝 Подготовка промпта для LLM...")
    details_container.text(
        "Генерация промпта на основе экспертного комментария")
    supabase = st.session_state['supabase']
    if st.session_state.model_type == "CUSTOM":
        response = supabase.storage.from_("PINN_LLM_STORAGE").download("loss/loss_dinn_custom.py")
    else:
        response = supabase.storage.from_("PINN_LLM_STORAGE").download("loss/loss_dinn_primary.py")

    loss_filepath = load_python_file_to_tmp(response)
    code = get_loss_func_as_str(loss_filepath)
    loss_before = code
    create_get_loss_based_on_recommendation_prompt(
        PROMPT_FILE_PATH, comment_class, comment_subclass, EXPERT_COMMENT, code)
    progress_bar.progress(10)
    time.sleep(0.5)

    # Шаг 2: Отправка запроса к LLM
    status_text.text("🤖 Отправка запроса к языковой модели...")
    details_container.text("Ожидание ответа от LLM")
    send_prompt(PROMPT_FILE_PATH, LLM_URL, client, ANSWER_FILE_PATH)
    progress_bar.progress(30)
    time.sleep(1)

    # Шаг 3: Обработка ответа
    status_text.text("🔍 Обработка ответа от LLM...")
    details_container.text("Извлечение кода из ответа")
    json_text = load_text_to_json(ANSWER_FILE_PATH)
    code = llm_answer_to_python_code(json_text)
    loss_after = code
    # print(code)
    if st.session_state.mode == 'DEV':
        details_container.text(f"Полученный код:\n```python\n{code}\n```")
    progress_bar.progress(40)
    # return

    # Шаг 4: Сохранение и первая проверка
    status_text.text("💾 Сохранение сгенерированного кода...")
    # save(LOSS_FILE_PATH, code)
    loss_file_path, content = save_py(LOSS_FILE_PATH, code)
    progress_bar.progress(50)
    # print(file_path)
    # return
    status_text.text("🧪 Первая проверка кода...")
    details_container.text("Запуск тестера для проверки корректности")
    # RUN_TESTER_COMMAND[2] = code
    file_path, content = create_file_in_tmp(code, LOSS_CHECK_FILE_NAME,
                                         LOSS_CHECK_START_FILE_PATH, LOSS_CHECK_END_FILE_PATH)
    output = subprocess.run(
        [f"{sys.executable}", file_path, code], capture_output=True)
    print(output.stdout)

    if "True" in str(output.stdout):
        t = (True, '')
        print(f"Результат: {t}")
    else:
        t = (False, str(output.stdout))
    is_correct = t[0]
    status_text.text(is_correct)
    print(is_correct)
    error = t[1]
    print(error)

    error_counter = 0
    max_error_iterations = 6

    # Шаг 5: Цикл исправления ошибок
    while not is_correct and error_counter < max_error_iterations:
        error_counter += 1
        status_text.text(f"⚠️ Исправление ошибок (попытка {error_counter})...")
        # details_container.text(f"Обнаружена ошибка: {error[:100]}...")
        if st.session_state.mode == 'DEV':
            error_container.error(f"Ошибка: {error}")

        progress_value = 50 + (error_counter * 10)
        progress_bar.progress(min(progress_value, 80))

        create_prompt_to_fix_error(PROMPT_FIX_ERROR_FILE_PATH, code, error)

        if error_counter % 3 == 0:
            details_container.text("Повторная отправка основного промпта...")
            send_prompt(PROMPT_FILE_PATH, LLM_URL, client, ANSWER_FILE_PATH)
            json_text = load_text_to_json(ANSWER_FILE_PATH)
        else:
            details_container.text(
                "Отправка промпта для исправления ошибки...")
            send_prompt(PROMPT_FIX_ERROR_FILE_PATH,
                        LLM_URL, client, ANSWER_FIX_ERROR_FILE_PATH)
            json_text = load_text_to_json(ANSWER_FIX_ERROR_FILE_PATH)

        code = llm_answer_to_python_code(json_text)
        loss_after = code
        loss_file_path, content = save_py(LOSS_FILE_PATH, code)

        # Проверка исправленного кода
        details_container.text("Проверка исправленного кода...")
        # RUN_TESTER_COMMAND[2] = code
        file_path, content = create_file_in_tmp(code, LOSS_CHECK_FILE_NAME,
                                             LOSS_CHECK_START_FILE_PATH, LOSS_CHECK_END_FILE_PATH)
        output = subprocess.run(
            [f"{sys.executable}", file_path, code], capture_output=True)
        if "True" in str(output.stdout):
            t = (True, '')
            print(f"Результат: {t}")
        else:
            t = (False, str(output.stdout))
        is_correct = t[0]
        status_text.text(is_correct)
        error = t[1]

        if is_correct:
            error_container.empty()
            details_container.text("✅ Ошибки исправлены!")
            progress_bar.progress(80)

    # return
    experiment_logger = TensorBoardExperimentLogger(log_dir="./runs/")
    experiment_logger.save_expert_comment(EXPERT_COMMENT)
    experiment_logger.save_loss_func(loss_before, loss_after)
    experiment_logger.save_LLM_model_name(LLM_MODEL_NAME)
    # Если превышено максимальное количество попыток
    if not is_correct:
        experiment_logger.save_loss_function_error_counter(error_counter=error_counter, is_fixed = False)
        status_text.text(
            "❌ Не удалось исправить ошибки после нескольких попыток")
        error_container.error(f"Последняя ошибка: {error}")
        progress_bar.progress(100)
        st.error("Процесс остановлен из-за непреодолимых ошибок. Пожалуйста, обновите страницу")
        return
    experiment_logger.save_loss_function_error_counter(error_counter=error_counter, is_fixed = True)
    # Шаг 6: Запуск обучения PINN
    status_text.text("🏃‍♂️ Запуск обучения PINN модели...")
    details_container.text(
        "Обучение нейросети (это может занять некоторое время)")
    progress_bar.progress(90)

    # Запуск с индикатором прогресса
    filename = ""
    with st.spinner("Идет обучение модели..."):
        file_path, content = create_file_in_tmp(code, PINN_NEW_FILE_NAME,
                                             PINN_RUN_START_FILE_PATH, PINN_RUN_END_FILE_PATH)
        output = subprocess.run(
            [f"{sys.executable}", file_path], capture_output=True, text=True)
        print(output.stdout)
        text = str(output.stdout)
        lines = text.strip().split('\n')
        last_line = lines[-1] if lines else ""
        filename = last_line.split("Model saved to ")[-1].strip()
        print(filename)

    # Шаг 7: Завершение
    status_text.text("✅ Обучение завершено!")
    # details_container.text("Процесс генерации и обучения успешно завершен")
    progress_bar.progress(100)

    # Показ результатов
    st.success("Модель успешно сгенерирована и обучена!")

    if st.session_state.mode == 'DEV':
    # Дополнительная информация о результате
        with st.expander("Детали выполнения"):
            st.text("Логи выполнения:")
            st.code(output.stdout[:1000] +
                    "..." if len(output.stdout) > 1000 else output.stdout)

            if output.stderr:
                st.warning("Предупреждения/ошибки:")
                st.code(output.stderr)
    supabase = st.session_state['supabase']

    response = supabase.storage.from_("PINN_LLM_STORAGE").download("data.csv")
    filepath = load_data_to_tmp(response)
    timesteps, susceptible, infected, dead, recovered, x = get_data_for_model(filepath)
    if st.session_state.model_type == "CUSTOM":
        response = supabase.storage.from_("PINN_LLM_STORAGE").download("NEW_MODEL_dinn_cuda_2.pth")
    else:
        response = supabase.storage.from_("PINN_LLM_STORAGE").download("dinn_cuda_03_10.pth")

    
    filepath = load_model_to_tmp(response)
    train_size = 180
    loaded_dinn = load_model(filepath,
                             timesteps, susceptible, infected, dead, recovered, train_size)
    
    loaded_dinn_new = load_model(filename,
                                 timesteps, susceptible, infected, dead, recovered, train_size)
    
    
    experiment_logger.save_model(loaded_dinn_new)
    
    S_pred, I_pred, D_pred, R_pred, alpha_pred = loaded_dinn.predict()

    S_pred_new, I_pred_new, D_pred_new, R_pred_new, alpha_pred_new = loaded_dinn_new.predict()


    st.title("Сравнение моделей")
    col1, col2 = st.columns(2)
    OLD_MODEL_NAME = "PINN"
    NEW_MODEL_NAME = "NEW_PINN"
    with col1:
        with st.expander("📈 Susceptible (S) - Развернуть/Свернуть", expanded=True):
            fig_s = plot_S_comparison(timesteps, x, susceptible, S_pred, S_pred_new)
            st.plotly_chart(fig_s)
            st.write("📊 Метрики моделей для S")
            metrics_S = calculate_metrics(susceptible[x:x+30], S_pred[x:x+30])
            metrics_SS = calculate_metrics(susceptible[x:x+30], S_pred_new[x:x+30])
            comparison_table = compare_metrics(metrics_S, metrics_SS, OLD_MODEL_NAME, NEW_MODEL_NAME)

    with col2:
        with st.expander("🦠 Infected (I) - Развернуть/Свернуть", expanded=True):
            fig_i = plot_I_comparison(timesteps, x, infected, I_pred, I_pred_new)
            st.plotly_chart(fig_i)
            st.write("📊 Метрики моделей для I")
            metrics_I = calculate_metrics(infected[x:x+30], I_pred[x:x+30])
            metrics_II = calculate_metrics(infected[x:x+30], I_pred_new[x:x+30])
            comparison_table = compare_metrics(metrics_I, metrics_II, OLD_MODEL_NAME, NEW_MODEL_NAME)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("💊 Recovered (R) - Развернуть/Свернуть", expanded=True):
            fig_r = plot_R_comparison(timesteps, x, recovered, R_pred, R_pred_new)
            st.plotly_chart(fig_r)
            st.write("📊 Метрики моделей для R")
            metrics_R = calculate_metrics(recovered[x:x+30], R_pred[x:x+30])
            metrics_RR = calculate_metrics(recovered[x:x+30], R_pred_new[x:x+30])
            comparison_table = compare_metrics(metrics_R, metrics_RR, OLD_MODEL_NAME, NEW_MODEL_NAME)

    with col2:
        with st.expander("⚰️ Dead (D) - Развернуть/Свернуть", expanded=True):
            fig_d = plot_D_comparison(timesteps, x, dead, D_pred, D_pred_new)
            st.plotly_chart(fig_d)
            st.write("📊 Метрики моделей для D")
            metrics_D = calculate_metrics(dead[x:x+30], D_pred[x:x+30])
            metrics_DD = calculate_metrics(dead[x:x+30], D_pred_new[x:x+30])
            comparison_table = compare_metrics(metrics_D, metrics_DD, OLD_MODEL_NAME, NEW_MODEL_NAME)
    
    experiment_logger.save_metrics(metrics_SS, metrics_II, metrics_RR, metrics_DD)
    experiment_logger.save_comparing_graph(fig_s, fig_i, fig_r, fig_d)
    

    st.subheader("Эпид.параметры")
    r0_value = get_R0(S_pred_new, I_pred_new, R_pred_new, D_pred_new, timesteps).numpy()
    st.metric("R0 (basic reproduction number)", f"{r0_value:.3f}")
        

    st.plotly_chart(display_compared_epid_params(S_pred, I_pred, R_pred, D_pred, timesteps, S_pred_new, I_pred_new, R_pred_new, D_pred_new, timesteps), width='stretch')

    # st.write("Если новый проноз Вас устраивает больше предыдущего, то нажмите на кнопку сохранения. После с новой сохраненной моделью можно будет работать, для этого нужно в левой боковой панели (настройки) главной страницы выбрать Кастомную модель, а не LTS Модель.")
    # st.write("Если Вы недовольны новым пронозом, то вернитесь на главную страницу и попытайтесь конкретнее сформулировать свое указание модели.")
    
    with st.expander("💡 Что делать после получения прогноза?", expanded=True):
        st.success("✅ **Если прогноз устраивает:**")
        st.markdown("""
        1. Нажмите кнопку **«Сохранить модель в Storage»**
        2. Вы автоматически перейдете на главную страницу
        3. В боковой панели выберите **«Кастомизированная модель»** для работы с сохраненной моделью
        """)
        
        st.warning("❌ **Если прогноз не устраивает:**")
        st.markdown("""
        1. Вернитесь на главную страницу
        2. Переформулируйте указания для модели
        """)
    # Кнопки возврата и сохранения
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("↩️ Вернуться на главную страницу", 
             use_container_width=True,
             on_click=lambda: (
                 setattr(st.session_state, 'current_page', 'main')
             )):
            pass
            # return
    st.session_state.current_model_path = filename
    st.session_state.current_loss_path = loss_file_path
    with col2:
        st.button(
            "💾 Сохранить модель в Storage",
            use_container_width=True,
            on_click=save_model_callback,
            key="save_model_btn"
        )

    if 'save_status' in st.session_state:
        st.write(st.session_state['save_status'])
            
    if st.session_state.mode == 'DEV':
        download_temp_file(loss_file_path)
        download_temp_file(filename)

    st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlock"]) {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #f0f6c1;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

def save_model_callback():
    try:
        supabase = st.session_state['supabase']
        filename = st.session_state.current_model_path
        loss_file_path = st.session_state.get('current_loss_path')

        with open(filename, "rb") as model_file:
            response = (
                supabase.storage
                .from_("PINN_LLM_STORAGE")
                .upload(
                    file=model_file,
                    path="NEW_MODEL_dinn_cuda_2.pth",
                    file_options={"cache-control": "3600", "upsert": "true"}
                )
            )

        with open(loss_file_path, "rb") as loss_file:
            response = (
                supabase.storage
                .from_("PINN_LLM_STORAGE")
                .upload(
                    file=loss_file,
                    path="loss/loss_dinn_custom.py",
                    file_options={"cache-control": "3600", "upsert": "true"}
                )
            )

        st.success("✅ Модель успешно сохранена!")
        st.session_state.current_page = "main"
        # Можно не менять current_page, тогда страница не "переедет"
    except Exception as e:
        st.error(f"❌ Ошибка при сохранении модели: {str(e)}")