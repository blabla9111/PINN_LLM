import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from web.backend.utils import *

def start_page():
    
    timesteps, susceptible, infected, dead, recovered, x = get_data_for_model(
        "data.csv")
    loaded_dinn = load_model('./saved_models/dinn_1.pth',
                             timesteps, susceptible, infected, dead, recovered)
    S_pred, I_pred, D_pred, R_pred, alpha_pred = loaded_dinn.predict()
    st.title("Информация о модели")

    # Разделение на две колонки
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈Графики")

        # S
        fig = plot_sidr_predictions(
            timesteps=timesteps,
            x=x,
            susceptible=susceptible,  
            infected=infected,        
            dead=dead,        
            recovered=recovered,      
            S_pred=S_pred,
            I_pred=I_pred,
            D_pred=D_pred,
            R_pred=R_pred
        )



        # Отображение в Streamlit
        st.pyplot(fig)

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

        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.subheader("Эпид.параметры")
        st.metric("R0 (basic reproduction number)",
                  get_R0(S_pred, I_pred, D_pred, R_pred, timesteps))

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

                # Сохраняем результаты в session state
                st.session_state.comment_analysis = "analysis_result"
                st.session_state.user_comment = translate_to_en(comment)

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
