import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from web.backend.utils import *
from lib.create_file import *



def start_page():
    supabase = st.session_state['supabase']
    
    response = supabase.storage.from_("PINN_LLM_STORAGE").download("data.csv")
    filepath = load_data_to_tmp(response)

    timesteps, susceptible, infected, dead, recovered, x = get_data_for_model(filepath)
    

    # Элементы в боковой панели
    st.sidebar.title("Настройки")
    selected_model = st.sidebar.selectbox("Выберите модель", ["LTS Модель (по умолчанию)", "Кастомизированная модель"])

    if selected_model == "Кастомизированная модель":
        response = supabase.storage.from_("PINN_LLM_STORAGE").download("NEW_MODEL_dinn_cuda_2.pth")
        st.session_state.model_type = "CUSTOM"
    else:
        response = supabase.storage.from_("PINN_LLM_STORAGE").download("dinn_cuda_03_10.pth")
        st.session_state.model_type = "LTS"

    selected_model = st.sidebar.selectbox("Выберите режим работы", ["Пользователь (по умолчанию)", "Разработчик"])

    if selected_model == "Разработчик":
        st.session_state.mode = "DEV"
    else:
        st.session_state.mode = "USER"
        # st.session_state.mode = "DEV"

    show_mode_indicator()

        
    filepath = load_model_to_tmp(response)
    train_size = 180
    loaded_dinn = load_model(filepath,
                             timesteps, susceptible, infected, dead, recovered, train_size)
    S_pred, I_pred, D_pred, R_pred, alpha_pred = loaded_dinn.predict()


    st.title("Информация о модели и прогноз")

    # Разделение на две колонки
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈Графики")

        # S
        fig = plot_sidr_predictions_plotly(
            timesteps=timesteps,
            x=x,
            susceptible=susceptible,
            infected=infected,
            dead=dead,
            recovered=recovered,
            S_pred=S_pred,
            I_pred=I_pred,
            D_pred=D_pred,
            R_pred=R_pred,
            start_day='2020-05-07'
        )

        # Отображение в Streamlit
        st.plotly_chart(fig)

    with col2:
        st.header("📊 Метрики моделей")

        # Вычисляем метрики для всех компонентов
        metrics_S = calculate_metrics(susceptible[x:x+30], S_pred[x:x+30])
        metrics_I = calculate_metrics(infected[x:x+30], I_pred[x:x+30])
        metrics_R = calculate_metrics(recovered[x:x+30], R_pred[x:x+30])
        metrics_D = calculate_metrics(dead[x:x+30], D_pred[x:x+30])

        # Создаем общую таблицу
        metrics_df = pd.DataFrame({
            'Метрика': list(metrics_I.keys()),
            'S (Susceptible)': list(metrics_S.values()),
            'I (Infected)': list(metrics_I.values()),
            'R (Recovered)': list(metrics_R.values()),
            'D (Dead)': list(metrics_D.values())
        })

        st.dataframe(metrics_df, hide_index=True, width='stretch')

        st.subheader("Эпид.параметры")
        r0_value = get_R0(S_pred, I_pred, R_pred, D_pred, timesteps).numpy()
        st.metric("R0 (basic reproduction number)", f"{r0_value:.3f}")
        

        st.plotly_chart(display_epid_params(S_pred, I_pred, R_pred, D_pred, timesteps), width='stretch')

        # Дополнительная статистика
        # st.subheader("Дополнительная информация")
        # st.metric("Объем данных", len(timesteps))
        # st.metric("Имя модели", "dinn_1")
        # st.metric("Версия модели", "v1.0.0")

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

                # Сохраняем результаты в session state
                st.session_state.comment_analysis = "analysis_result"
                st.session_state.comment_primary = comment
                st.session_state.user_comment = translate_to_en(comment)
                # print(st.session_state.user_comment)

                # Переключаемся на страницу результатов
                st.session_state.current_page = "results"
                st.rerun()
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
