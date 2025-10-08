import streamlit as st
import time
from web.backend.utils import *

from comment_classificator.match_loss_classification import predict_class_and_sub_class


CLASS_TYPE_INFO = {"1": ["Поведение эпидемической кривой не соответствует ожиданиям эксперта.", {
    "1": "График кол-ва инфицированных",
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

def insert_expert_comment(supabase, comment, class_num, subclass_num, approved=False):
    data = {
        "comment": comment,
        "class": class_num,
        "subclass": subclass_num,
        "approved": approved
    }
    
    response = supabase.table("expert_comment").insert(data).execute()
    return response

def class_subclass_page():
    show_mode_indicator()
    st.title("Результаты анализа экспертных указаний")


    if 'comment_analysis' not in st.session_state:
        st.warning("Нет данных для анализа. Вернитесь на главную страницу.")
        if st.button("Вернуться на главную"):
            st.session_state.current_page = "main"
            st.rerun()
        return

    # analysis = st.session_state.comment_analysis
    comment = st.session_state.user_comment
    top_indices, top_probs, is_valid = predict_class_and_sub_class(comment)
    comment_class = str(top_indices[0])
    comment_subclass = str(top_indices[1])
    st.session_state.user_comment_class = comment_class
    st.session_state.user_comment_subclass = comment_subclass
    print(comment_class)
    print(comment_subclass)
    if is_valid==False:
            if st.session_state.mode == 'DEV':
                st.error("❌ Некорректная иерархия классов")
                st.warning("Предсказанные классы не образуют валидную пару класс-подкласс")
            st.write("⚠️ Перейдите, пожалуйста, на главную страницу... Нужно переформулировать комментарий (указание для модели), так как модель не смогла однозначно определить темы комментария", icon="⚠️")
    # else:        
    #     st.write("⚠️ Перейдите, пожалуйста, на главную страницу... Нужно переформулировать комментарий (указание для модели), так как модель не смогла однозначно определить темы комментария", icon="⚠️")
        # Отображение исходного комментария
    st.subheader("Ваш комментарий (указание для модели):")
    st.info(f'"{st.session_state.comment_primary}"')
    supabase = st.session_state['supabase']
    response = insert_expert_comment(supabase, comment=st.session_state.comment_primary,
                                        class_num=comment_class, 
                                        subclass_num=comment_subclass,
                                        approved=False)
        # st.write()
    st.session_state.comment_id = response.data[0]['id']
        # Разделение на колонки для результатов
    col1 = st.columns(1)

    with col1[0]:
            st.subheader("Программа определила, что Ваш комментарий относится к следующим темам:")
            st.info(f"**{CLASS_TYPE_INFO[comment_class][0]}**", icon="📋")
            st.info(f"**{CLASS_TYPE_INFO[comment_class][1][comment_subclass]}**", icon="📋")

            if st.session_state.mode == 'DEV':
                # Дополнительная информация о результате
                with st.expander("Детали выполнения"):
                    st.write(f"**🎯 Основной класс:** {comment_class}")
                    st.caption(CLASS_TYPE_INFO[comment_class][0])
                    st.write(f"Уверенность в классе: {round(top_probs[0] * 100)}%")
                    
                    st.write(f"**🔍 Подкласс:** {comment_subclass}")
                    st.caption(CLASS_TYPE_INFO[comment_class][1][comment_subclass])
                    st.write(f"Уверенность в подклассе: {round(top_probs[1] * 100)}%")

            # Время анализа
            # st.write(f"**Время анализа:** {datetime.now().strftime(" % Y-%m-%d % H: % M: % S")}")

    st.subheader("Подходят ли предложенные темы под ваш комментарий?")
    st.write("После подтверждения программа попытается адаптировать прогноз по Вашему комментарию")
        # st.write("Подходят ли предложенные класс и подкласс под ваш комментарий?")

    col_confirm1, col_confirm2 = st.columns(2)

    with col_confirm1:
            if st.button("✅ Да, подходят", 
                type="primary", 
                width='stretch',
                on_click=confirm_classification_callback ):
                pass

    with col_confirm2:
            if st.button("❌ Нет, не подходят", 
                type="secondary", 
                width='stretch',
                on_click=reject_classification_callback):
                pass

        # Разделитель
    st.divider()

    # Кнопки для навигации
    col_btn1, col_btn2 = st.columns(2)

    # with col_btn1:
    #     st.button("Вернуться на главную", 
    #          type="primary", 
    #          on_click=return_to_main_callback)
    with col_btn1:
        st.button("↩️ Вернуться на главную страницу", 
             use_container_width=True,
             on_click=return_to_main_callback)

def return_to_main_callback():
    # Сохраняем в историю
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []

    st.session_state.comment_history.append({
        "comment": st.session_state.user_comment,
        "main_class": st.session_state.user_comment_class,
        "subclass": st.session_state.user_comment_subclass,
    })

    st.session_state.current_page = "main"

def confirm_classification_callback():
    # Сохраняем в историю
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []

    st.session_state.comment_history.append({
        "comment": st.session_state.user_comment,
        "main_class": st.session_state.user_comment_class,
        "subclass": st.session_state.user_comment_subclass,
        "confirmed": True,
    })
    supabase = st.session_state['supabase']
    update_response = supabase.table("expert_comment").update({"approved": True}).eq("id", st.session_state.comment_id).execute()
    st.session_state.current_page = "generate new model"


def reject_classification_callback():
    # Сохраняем в историю как неподтвержденный
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []

    st.session_state.comment_history.append({
        "comment": st.session_state.user_comment,
        "main_class": st.session_state.user_comment_class,
        "subclass": st.session_state.user_comment_subclass,
        "confirmed": False,
    })

    st.toast("⚠️ Переход на главную страницу... Нужно переформулировать комментарий (указание для модели)", icon="⚠️")
    st.session_state.current_page = "main"

