from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
from PINN_class import DINN
from comment_classificator.match_loss_classification import predict_class_and_sub_class
from lib.prompt_sender import *
from lib.parser import *
from lib.loss_update import save
import subprocess
import time
from huggingface_hub import InferenceClient


client = InferenceClient(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                         token=st.secrets['HUGGINGFACE_HUB_TOKEN'])


CLASS_TYPE_INFO = {"1": ["Поведение эпидемической кривой не соответствует ожиданиям эксперта.",{
    "1": "График кол-ва инфицированнных",
    "2": "График кол-ва восприимчивых",
    "3": "График кол-ва выздоровевших",
    "4": "График кол-ва скончавшихся"
}],
    "2": ["Учитывание мер противодействия распространению эпидемии.", {
        "1": "Временное несоответствие эффекта мер",
        "2": "Неверное отражение эффектов мер",
        "3": "Отсутствие отражения эффектов мер",
}],
    "3": ["Прогноз содержит неверные параметры или временную динамику эпидемиологического процесса.", {
        "1": "Несоответствие по времени",
        "2": "Несоответствие по кол-ву",
}],
    "4": ["Позиция пика эпидемии не соответствует ожиданиям эксперта.", {
        "1": "Ранний/поздний пик",
        "2": "Неверное кол-во заболевших во время пика",
}]}

# Настройка страницы
st.set_page_config(page_title="Анализ модели", layout="wide")


def load_model(filepath, t, S_data, I_data, D_data, R_data):
    """Загрузить модель"""
    print("загрузка модели")
    checkpoint = torch.load(filepath)

    # Создаем экземпляр модели
    model = DINN(t, S_data, I_data, D_data, R_data)

    # Загружаем параметры
    model.load_state_dict(checkpoint['model_state_dict'])
    model.beta_tilda = checkpoint['beta_tilda']
    model.gamma_tilda = checkpoint['gamma_tilda']
    model.S_max = checkpoint['S_max']
    model.I_max = checkpoint['I_max']
    model.D_max = checkpoint['D_max']
    model.R_max = checkpoint['R_max']
    model.S_min = checkpoint['S_min']
    model.I_min = checkpoint['I_min']
    model.D_min = checkpoint['D_min']
    model.R_min = checkpoint['R_min']
    model.t = checkpoint['t']
    model.S = checkpoint['S']
    model.I = checkpoint['I']
    model.D = checkpoint['D']
    model.R = checkpoint['R']

    # Обновляем производные атрибуты
    model.t_float = model.t.float()
    model.t_batch = torch.reshape(model.t_float, (len(model.t), 1))
    model.S_hat = (model.S - model.S_min) / (model.S_max - model.S_min)
    model.I_hat = (model.I - model.I_min) / (model.I_max - model.I_min)
    model.D_hat = (model.D - model.D_min) / (model.D_max - model.D_min)
    model.R_hat = (model.R - model.R_min) / (model.R_max - model.R_min)

    print(f"Model loaded from {filepath}")
    return model


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics

    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values

    Returns:
    dict: Dictionary with MAE, MSE, RMSE, R2 metrics
    """
    # Convert to numpy arrays if they are tensors
    if torch.is_tensor(y_true):
        y_true = y_true.detach().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().numpy()

    # Ensure they are 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    return metrics

# Главная страница


def main_page():
    covid_cases = pd.read_csv('data.csv')


    S = covid_cases['S']
    I = covid_cases['I']
    D = covid_cases['D']
    R = covid_cases['R']
    susceptible = []
    infected = []
    dead = []
    recovered = []
    timesteps = []

    d1 = covid_cases['S']
    d2 = covid_cases['I']
    d3 = covid_cases['D']
    d4 = covid_cases['R']
    d5 = covid_cases['t']

    for item in range(len(d5)):
        if item % 1 == 0:
            susceptible.append(d1[item])
            infected.append(d2[item])
            dead.append(d3[item])
            recovered.append(d4[item])
            timesteps.append(d5[item])

    x = 180
    loaded_dinn = load_model('./saved_models/dinn_1.pth',
                             timesteps, susceptible, infected, dead, recovered)
    S_pred, I_pred, D_pred, R_pred, alpha_pred = loaded_dinn.predict()
    st.title("Информация о модели")

    # Разделение на две колонки
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈График I")
        fig, ax = plt.subplots(figsize=(10, 6))


        ax.scatter(timesteps[:x][::10], infected[:x][::10],
                c='blue', alpha=0.5, lw=0.5, label='Real data')

        ax.scatter(timesteps[x:][::10], infected[x:][::10],
                c='white', edgecolors='black', alpha=0.5, lw=0.5, label='Future data')

        ax.plot(timesteps, I_pred.detach().numpy(),
                'black', alpha=0.9, lw=2, label='Model', linestyle='dashed')

        ax.set_xlabel("Time, days")
        ax.set_ylabel("Infected, persons")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Отображение в Streamlit
        st.pyplot(fig)

    with col2:
        st.header("📊Метрики модели")
        metrics_I = calculate_metrics(infected[x:187], I_pred[x:187]) # на неделю только оцениваем
        metrics_df = pd.DataFrame({
            'Метрика': list(metrics_I.keys()),
            'Значение': list(metrics_I.values())
        })

        metrics_df = pd.DataFrame(metrics_df)
        st.dataframe(metrics_df, hide_index=True, width='stretch')

        # Дополнительная статистика
        st.subheader("Дополнительная информация")
        st.metric("Объем данных", len(timesteps))
        st.metric("Имя модели", "dinn_1")
        st.metric("Версия модели", "v1.0.0")

    # Разделитель
    st.divider()

    # Поле для ввода комментария и кнопка отправки
    st.header("Комментарии")

    comment = st.text_area(
        "Оставьте ваш комментарий, если в чем-то не согласны с прогнозом:",
        placeholder="Введите ваши наблюдения или предложения по прогнозу модели...",
        height=100,
        key="comment_input"
    )

    col_btn1, col_btn2 = st.columns([1, 6])

    with col_btn1:
        if st.button("Отправить", type="primary", key="submit_btn"):
            if comment.strip():
                # Анализируем комментарий
                # analysis_result = analyze_comment(comment)

                # Сохраняем результаты в session state
                st.session_state.comment_analysis = "analysis_result"
                st.session_state.user_comment = comment

                # Переключаемся на страницу результатов
                st.session_state.current_page = "results"
                st.rerun()
            else:
                st.warning("Пожалуйста, введите комментарий перед отправкой")

    with col_btn2:
        if st.button("Очистить", key="clear_btn"):
            st.rerun()

    # Дополнительная секция с историей комментариев
    with st.expander("История комментариев"):
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
            st.info("История комментариев пуста")

# Страница с результатами анализа


def results_page():
    st.title("Результаты анализа комментария")

    if 'comment_analysis' not in st.session_state:
        st.warning("Нет данных для анализа. Вернитесь на главную страницу.")
        if st.button("Вернуться на главную"):
            st.session_state.current_page = "main"
            st.rerun()
        return

    # analysis = st.session_state.comment_analysis
    comment = st.session_state.user_comment
    top_indices, top_probs = predict_class_and_sub_class(comment)
    comment_class = str(top_indices[0])
    comment_subclass = str(top_indices[1])
    st.session_state.user_comment_class = comment_class
    st.session_state.user_comment_subclass = comment_subclass
    print(comment_class)
    print(comment_subclass)
    # Отображение исходного комментария
    st.subheader("Ваш комментарий:")
    st.info(f'"{comment}"')

    # Разделение на колонки для результатов
    col1 = st.columns(1)

    with col1[0]:
        st.subheader("Классификация:")

        # Отображение основного класса
        st.metric("Основной класс", comment_class + " " +
                  CLASS_TYPE_INFO[comment_class][0])

        # Отображение подкласса
        st.metric("Подкласс", comment_subclass + " " +
                  CLASS_TYPE_INFO[comment_class][1][comment_subclass])

        # Уверенность модели
        st.metric("Уверенность модели в классе", f"{top_probs[0] * 100}%")

        st.metric("Уверенность модели в подклассе", f"{top_probs[1] * 100}%")

        # Время анализа
        # st.write(f"**Время анализа:** {datetime.now().strftime(" % Y-%m-%d % H: % M: % S")}")

    # with col2:
        # st.subheader("Визуализация:")

        # # Простая круговая диаграмма для уверенности
        # fig, ax = plt.subplots(figsize=(6, 6))
        # sizes = [top_probs[0], 1 - top_probs[0]]
        # labels = ['Уверенность', 'Неуверенность']
        # colors = ['#4CAF50', '#F44336']

        # ax.pie(sizes, labels=labels, colors=colors,
        #        autopct='%1.1f%%', startangle=90)
        # ax.axis('equal')
        # ax.set_title('Уверенность классификации')

        # st.pyplot(fig)

    # Дополнительная информация
    # st.divider()
    # st.subheader("Рекомендации:")

    # # Рекомендации в зависимости от класса
    # if analysis['main_class'] == "Проблема":
    #     st.warning("""
    #     **Рекомендуемые действия:**
    #     - Сообщите о проблеме технической поддержке
    #     - Укажите подробности воспроизведения ошибки
    #     - Приложите скриншоты или логи ошибок
    #     """)
    # elif analysis['main_class'] == "Предложение":
    #     st.success("""
    #     **Рекомендуемые действия:**
    #     - Ваше предложение будет рассмотрено командой разработки
    #     - Ожидайте обратной связи в течение 3 рабочих дней
    #     - Благодарим за участие в улучшении продукта!
    #     """)
    # elif analysis['main_class'] == "Вопрос":
    #     st.info("""
    #     **Рекомендуемые действия:**
    #     - Ответ на ваш вопрос будет предоставлен в течение 24 часов
    #     - Проверьте раздел FAQ на наличие похожих вопросов
    #     - Свяжитесь с технической поддержкой для срочных вопросов
    #     """)
    # else:
    #     st.success("""
    #     **Благодарим за отзыв!**
    #     - Ваше мнение очень важно для нас
    #     - Мы постоянно работаем над улучшением продукта
    #     """)

    # Кнопки для навигации
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Вернуться на главную", type="primary"):
            # Сохраняем в историю
            if 'comment_history' not in st.session_state:
                st.session_state.comment_history = []

            st.session_state.comment_history.append({
                "comment": comment,
                "main_class": comment_class,
                "subclass": comment_subclass,
                # "confidence": analysis['confidence'],
                # "timestamp": analysis['timestamp']
            })

            st.session_state.current_page = "main"
            st.rerun()

    with col_btn2:
        if st.button("Сделать новый прогноз с учетом комментария"):
            st.session_state.current_page = "generate new model"
            st.rerun()


def generate_model_page():
    PROMPT_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt.json'
    ANSWER_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt_answer.json'
    LOSS_FILE_PATH = 'loss_dinn_LLM.py'
    LOSS_PRIMARY_FILE_PATH = 'loss_dinn_primary.py'
    LLM_URL = 'http://localhost:1234/v1/chat/completions'
    RUN_PINN_COMMAND = ['python', 'PINN.py']
    RUN_TESTER_COMMAND = ['python', 'loss_dinn_check.py ', '']
    PROMPT_FIX_ERROR_FILE_PATH = "prompt_fix_error.json"
    ANSWER_FIX_ERROR_FILE_PATH = 'answer_fix_error_from_LLM_2.json'

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
    send_prompt(PROMPT_FILE_PATH, LLM_URL,client, ANSWER_FILE_PATH)
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
    save(LOSS_FILE_PATH, code)
    progress_bar.progress(50)
    return 
    status_text.text("🧪 Первая проверка кода...")
    details_container.text("Запуск тестера для проверки корректности")
    RUN_TESTER_COMMAND[2] = code
    output = subprocess.run(RUN_TESTER_COMMAND, capture_output=True, text=True)
    print(output.stdout)

    if "True" in output.stdout:
        t = (True, '')
        print(f"Результат: {t}")
    else:
        t = (False, output.stdout)
    # return
    is_correct = t[0]
    status_text.text(is_correct)
    print(is_correct)
    error = t[1]

    error_counter = 0
    max_error_iterations = 2

    # Шаг 5: Цикл исправления ошибок
    while not is_correct and error_counter < max_error_iterations:
        error_counter += 1
        status_text.text(f"⚠️ Исправление ошибок (попытка {error_counter})...")
        details_container.text(f"Обнаружена ошибка: {error[:100]}...")
        error_container.error(f"Ошибка: {error}")

        progress_value = 50 + (error_counter * 10)
        progress_bar.progress(min(progress_value, 80))

        create_prompt_to_fix_error(PROMPT_FIX_ERROR_FILE_PATH, code, error)

        if error_counter % 3 == 0:
            details_container.text("Повторная отправка основного промпта...")
            send_prompt(PROMPT_FILE_PATH, LLM_URL,client, ANSWER_FILE_PATH)
            json_text = load_text_to_json(ANSWER_FILE_PATH)
        else:
            details_container.text(
                "Отправка промпта для исправления ошибки...")
            send_prompt(PROMPT_FIX_ERROR_FILE_PATH,
                        LLM_URL, client,ANSWER_FIX_ERROR_FILE_PATH)
            json_text = load_text_to_json(ANSWER_FIX_ERROR_FILE_PATH)

        code = llm_answer_to_python_code(json_text)
        save(LOSS_FILE_PATH, code)

        # Проверка исправленного кода
        details_container.text("Проверка исправленного кода...")
        RUN_TESTER_COMMAND[2] = code
        output = subprocess.run(
            RUN_TESTER_COMMAND, capture_output=True, text=True, shell=True)
        # print(f'!!!!!!!!\n\n{output}')
        # return
        # t = eval(output.stdout)
        if "True" in output.stdout:
            t = (True, '')
            print(f"Результат: {t}")
        else:
            t = (False, output.stdout)
        is_correct = t[0]
        status_text.text(is_correct)
        error = t[1]

        if is_correct:
            error_container.empty()
            details_container.text("✅ Ошибки исправлены!")
            progress_bar.progress(80)

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
    with st.spinner("Идет обучение модели..."):
        output = subprocess.run(
            RUN_PINN_COMMAND, capture_output=True, text=True)

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
    covid_cases = pd.read_csv('data.csv')

    S = covid_cases['S']
    I = covid_cases['I']
    D = covid_cases['D']
    R = covid_cases['R']
    susceptible = []
    infected = []
    dead = []
    recovered = []
    timesteps = []

    d1 = covid_cases['S']
    d2 = covid_cases['I']
    d3 = covid_cases['D']
    d4 = covid_cases['R']
    d5 = covid_cases['t']

    for item in range(len(d5)):
        if item % 1 == 0:
            susceptible.append(d1[item])
            infected.append(d2[item])
            dead.append(d3[item])
            recovered.append(d4[item])
            timesteps.append(d5[item])
    x = 180
    loaded_dinn = load_model('./saved_models/dinn_1.pth',
                             timesteps, susceptible, infected, dead, recovered)
    loaded_dinn_new = load_model('./saved_models/NEW_MODEL_dinn_1.pth',
                             timesteps, susceptible, infected, dead, recovered)
    S_pred, I_pred, D_pred, R_pred, alpha_pred = loaded_dinn.predict()

    S_pred_new, I_pred_new, D_pred_new, R_pred_new, alpha_pred_new = loaded_dinn_new.predict()
    st.title("Информация о модели")

    # Разделение на две колонки
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈График I")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(timesteps[:x][::10], infected[:x][::10],
                   c='blue', alpha=0.5, lw=0.5, label='Real data')

        ax.scatter(timesteps[x:][::10], infected[x:][::10],
                   c='white', edgecolors='black', alpha=0.5, lw=0.5, label='Future data')

        ax.plot(timesteps, I_pred.detach().numpy(),
                'black', alpha=0.9, lw=2, label='Model', linestyle='dashed')
        ax.plot(timesteps, I_pred_new.detach().numpy(),
                'red', alpha=0.9, lw=2, label='NEW_Model', linestyle='dashed')

        ax.set_xlabel("Time, days")
        ax.set_ylabel("Infected, persons")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Отображение в Streamlit
        st.pyplot(fig)

# Инициализация session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

# Отображение текущей страницы
if st.session_state.current_page == "main":
    main_page()
elif st.session_state.current_page == "results":
    results_page()
elif st.session_state.current_page == "generate new model":
    generate_model_page()

# Стилизация
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
    .stInfo {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1890ff;
    }
</style>
""", unsafe_allow_html=True)
