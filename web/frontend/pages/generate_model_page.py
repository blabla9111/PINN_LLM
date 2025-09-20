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

client = InferenceClient(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                         token=st.secrets['HUGGINGFACE_HUB_TOKEN'])


def generate_model_page():
    
    PROMPT_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt.json'
    ANSWER_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt_answer.json'
    LOSS_FILE_PATH = 'loss_dinn_LLM.py'
    LOSS_PRIMARY_FILE_PATH = 'web/backend/PINN_utils/loss_dinn_primary.py'
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
    code = get_loss_func_as_str(LOSS_PRIMARY_FILE_PATH)
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
    # print(code)
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
    # output = subprocess.run(RUN_TESTER_COMMAND, capture_output=True, text=True)
    output = subprocess.run(
        [f"{sys.executable}", file_path, code], capture_output=True)
    print(output.stdout)
    # t =  eval(output.stdout)
    # return

    if "True" in str(output.stdout):
        t = (True, '')
        print(f"Результат: {t}")
    else:
        t = (False, str(output.stdout))
    # return
    is_correct = t[0]
    status_text.text(is_correct)
    print(is_correct)
    error = t[1]
    print(error)

    error_counter = 0
    max_error_iterations = 4

    # Шаг 5: Цикл исправления ошибок
    while not is_correct and error_counter < max_error_iterations:
        error_counter += 1
        status_text.text(f"⚠️ Исправление ошибок (попытка {error_counter})...")
        # details_container.text(f"Обнаружена ошибка: {error[:100]}...")
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
        loss_file_path, content = save_py(LOSS_FILE_PATH, code)

        # Проверка исправленного кода
        details_container.text("Проверка исправленного кода...")
        # RUN_TESTER_COMMAND[2] = code
        file_path, content = create_file_in_tmp(code, LOSS_CHECK_FILE_NAME,
                                             LOSS_CHECK_START_FILE_PATH, LOSS_CHECK_END_FILE_PATH)
        output = subprocess.run(
            [f"{sys.executable}", file_path, code], capture_output=True)
        # output = subprocess.run(
        #     RUN_TESTER_COMMAND, capture_output=True, text=True, shell=True)
        # print(f'!!!!!!!!\n\n{output}')
        # return
        # t = eval(output.stdout)
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
    # Если превышено максимальное количество попыток
    if not is_correct:
        status_text.text(
            "❌ Не удалось исправить ошибки после нескольких попыток")
        error_container.error(f"Последняя ошибка: {error}")
        progress_bar.progress(100)
        st.error("Процесс остановлен из-за непреодолимых ошибок")
        return

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
    details_container.text("Процесс генерации и обучения успешно завершен")
    progress_bar.progress(100)

    # Показ результатов
    st.success("Модель успешно сгенерирована и обучена!")

    # Дополнительная информация о результате
    with st.expander("Детали выполнения"):
        st.text("Логи выполнения:")
        st.code(output.stdout[:1000] +
                "..." if len(output.stdout) > 1000 else output.stdout)

        if output.stderr:
            st.warning("Предупреждения/ошибки:")
            st.code(output.stderr)

    st.write("Generate complete!")
    # return
    timesteps, susceptible, infected, dead, recovered, x = get_data_for_model(
        "data.csv")
    loaded_dinn = load_model('./saved_models/dinn_cuda.pth',
                             timesteps, susceptible, infected, dead, recovered)
    loaded_dinn_new = load_model(filename,
                                 timesteps, susceptible, infected, dead, recovered)
    S_pred, I_pred, D_pred, R_pred, alpha_pred = loaded_dinn.predict()

    S_pred_new, I_pred_new, D_pred_new, R_pred_new, alpha_pred_new = loaded_dinn_new.predict()

    # Подготовка данных
    real_data = {
        'S': susceptible,
        'I': infected,
        'R': recovered,
        'D': dead,
    }

    pred_old = {
        'S': S_pred,
        'I': I_pred,
        'D': D_pred,
        'R': R_pred
    }

    pred_new = {
        'S': S_pred_new,
        'I': I_pred_new,
        'D': D_pred_new,
        'R': R_pred_new
    }

    st.title("Сравнение моделей")
    col1, col2 = st.columns(2)


    with col1:
        with st.container():

            fig_s = plot_S_comparison(
                timesteps, x, susceptible, S_pred, S_pred_new)
            st.pyplot(fig_s)
            st.header("📊 Метрики моделей для S")

            metrics_I = calculate_metrics(susceptible[x:x+30], S_pred[x:x+30])
            metrics_II = calculate_metrics(susceptible[x:x+30], S_pred_new[x:x+30])

            comparison_table = compare_metrics(
                metrics_I, metrics_II, "PINN", "NEW_PINN")

        

    with col2:
        with st.container():
            fig_i = plot_I_comparison(
                timesteps, x, infected, I_pred, I_pred_new)
            st.pyplot(fig_i)
            st.header("📊 Метрики моделей для I")
            metrics_I = calculate_metrics(infected[x:x+30], I_pred[x:x+30])
            metrics_II = calculate_metrics(
                infected[x:x+30], I_pred_new[x:x+30])  # метрики второй модели

            # Создаем таблицу сравнения
            comparison_table = compare_metrics(
                metrics_I, metrics_II, "PINN", "NEW_PINN")
        

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            fig_r = plot_R_comparison(
                timesteps, x, recovered, R_pred, R_pred_new)
            st.pyplot(fig_r)
            st.header("📊 Метрики моделей для R")
            metrics_I = calculate_metrics(recovered[x:x+30], R_pred[x:x+30])
            metrics_II = calculate_metrics(
                recovered[x:x+30], R_pred_new[x:x+30])  # метрики второй модели

            # Создаем таблицу сравнения
            comparison_table = compare_metrics(
                metrics_I, metrics_II, "PINN", "NEW_PINN")

    with col2:
        with st.container():
            fig_d = plot_D_comparison(
                timesteps, x, dead, D_pred, D_pred_new)
            st.pyplot(fig_d)
            st.header("📊 Метрики моделей для D")
            metrics_I = calculate_metrics(dead[x:x+30], D_pred[x:x+30])
            metrics_II = calculate_metrics(
                dead[x:x+30], D_pred_new[x:x+30])  # метрики второй модели

            # Создаем таблицу сравнения
            comparison_table = compare_metrics(
                metrics_I, metrics_II, "PINN", "NEW_PINN")
            
    st.subheader("Эпид.параметры")
    st.metric("R0 (basic reproduction number)",
                  get_R0(S_pred, I_pred, R_pred, D_pred, timesteps))
        

    st.plotly_chart(display_compared_epid_params(S_pred, I_pred, R_pred, D_pred, timesteps, S_pred_new, I_pred_new, R_pred_new, D_pred_new, timesteps), width='stretch')

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



