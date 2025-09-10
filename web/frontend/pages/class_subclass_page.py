import streamlit as st

from comment_classificator.match_loss_classification import predict_class_and_sub_class

CLASS_TYPE_INFO = {"1": ["Поведение эпидемической кривой не соответствует ожиданиям эксперта.", {
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

def class_subclass_page():
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
