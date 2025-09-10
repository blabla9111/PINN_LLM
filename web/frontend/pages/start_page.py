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
        # на неделю только оцениваем
        metrics_I = calculate_metrics(infected[x:187], I_pred[x:187])
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
