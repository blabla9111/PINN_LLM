import streamlit as st
from web.backend.controllers import AnalysisController
from web.backend.utils.translator import translate

CLASS_TYPE_INFO = {
    "1": ["Поведение эпидемической кривой не соответствует ожиданиям эксперта.", {
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
    }]
}

def class_subclass_page():
    show_mode_indicator()
    st.title(translate("Результаты анализа экспертных указаний"))

    # Получаем конфиг и клиент из session state
    config = st.session_state['app_config']
    supabase = st.session_state['supabase']
    
    # Инициализация контроллера
    analysis_controller = AnalysisController(config)

    # Проверка наличия данных для анализа
    if 'comment_analysis' not in st.session_state:
        st.warning(translate("Нет данных для анализа. Вернитесь на главную страницу."))
        if st.button(translate("Вернуться на главную")):
            st.session_state.current_page = "main"
            st.rerun()
        return

    # Анализ комментария через контроллер
    comment = st.session_state.user_comment
    analysis_result = analysis_controller.analyze_comment(comment)
    
    if not analysis_result['success']:
        st.error(translate(f"Ошибка при анализе комментария: {analysis_result['error']}"))
        return

    # Извлекаем результаты анализа
    comment_class = analysis_result['comment_class']
    comment_subclass = analysis_result['comment_subclass']
    is_valid = analysis_result['is_valid']
    probabilities = analysis_result['probabilities']
    
    # Сохраняем в session state
    st.session_state.user_comment_class = comment_class
    st.session_state.user_comment_subclass = comment_subclass

    # Проверка валидности классификации
    if not is_valid:
        if st.session_state.mode == 'DEV':
            st.error(translate("❌ Некорректная иерархия классов"))
            st.warning(translate("Предсказанные классы не образуют валидную пару класс-подкласс"))
        st.write(translate("⚠️ Перейдите, пожалуйста, на главную страницу... Нужно переформулировать комментарий (указание для модели), так как модель не смогла однозначно определить темы комментария"), icon="⚠️")
    
    # Отображение исходного комментария
    st.subheader(translate("Ваш комментарий (указание для модели):"))
    st.info(f'"{st.session_state.comment_primary}"')
    
    # Сохранение экспертного комментария через контроллер
    save_result = analysis_controller.save_expert_comment(
        supabase, 
        comment=st.session_state.comment_primary,
        class_num=comment_class, 
        subclass_num=comment_subclass,
        approved=False
    )
    
    if save_result['success']:
        st.session_state.comment_id = save_result['comment_id']
    else:
        st.error(translate(f"Ошибка при сохранении комментария: {save_result['error']}"))

    # Разделение на колонки для результатов
    col1 = st.columns(1)

    with col1[0]:
        st.subheader(translate("Программа определила, что Ваш комментарий относится к следующим темам:"))
        st.info(f"**{translate(CLASS_TYPE_INFO[comment_class][0])}**", icon="📋")
        st.info(f"**{translate(CLASS_TYPE_INFO[comment_class][1][comment_subclass])}**", icon="📋")

        if st.session_state.mode == 'DEV':
            # Дополнительная информация о результате
            with st.expander(translate("Детали выполнения")):
                st.write(f"**{translate('🎯 Основной класс')}:** {comment_class}")
                st.caption(translate(CLASS_TYPE_INFO[comment_class][0]))
                st.write(f"{translate('Уверенность в классе')}: {analysis_result['confidence_class']}%")
                
                st.write(f"**{translate('🔍 Подкласс')}:** {comment_subclass}")
                st.caption(translate(CLASS_TYPE_INFO[comment_class][1][comment_subclass]))
                st.write(f"{translate('Уверенность в подклассе')}: {analysis_result['confidence_subclass']}%")
                
                # Информация о конфигурации
                st.write(f"**{translate('⚙️ Конфигурация')}:**")
                st.json({
                    "app_mode": config.app.MODE,
                    "debug": config.app.DEBUG,
                    "llm_model": config.llm.MODEL_NAME
                })

    st.subheader(translate("Подходят ли предложенные темы под ваш комментарий?"))
    st.write(translate("После подтверждения программа попытается адаптировать прогноз по Вашему комментарию"))

    col_confirm1, col_confirm2 = st.columns(2)

    with col_confirm1:
        if st.button(translate("✅ Да, подходят"), 
            type="primary", 
            use_container_width=True,
            on_click=lambda: confirm_classification_callback(supabase, analysis_controller)):
            pass

    with col_confirm2:
        if st.button(translate("❌ Нет, не подходят"), 
            type="secondary", 
            use_container_width=True,
            on_click=reject_classification_callback):
            pass

    # Разделитель
    st.divider()

    # Кнопки для навигации
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        st.button(translate("↩️ Вернуться на главную страницу"), 
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

def confirm_classification_callback(supabase, analysis_controller):
    # Сохраняем в историю
    if 'comment_history' not in st.session_state:
        st.session_state.comment_history = []

    st.session_state.comment_history.append({
        "comment": st.session_state.user_comment,
        "main_class": st.session_state.user_comment_class,
        "subclass": st.session_state.user_comment_subclass,
        "confirmed": True,
    })
    
    # Обновление статуса подтверждения через контроллер
    if hasattr(st.session_state, 'comment_id'):
        update_result = analysis_controller.update_comment_approval(
            supabase, 
            st.session_state.comment_id, 
            True
        )
        
        if not update_result['success']:
            st.error(translate(f"Ошибка при обновлении статуса комментария: {update_result['error']}"))
    
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

    st.toast(translate("⚠️ Переход на главную страницу... Нужно переформулировать комментарий (указание для модели)"), icon="⚠️")
    st.session_state.current_page = "main"

# Вспомогательная функция для отображения индикатора режима
def show_mode_indicator():
    if st.session_state.mode == 'DEV':
        st.sidebar.info(translate("🔧 Режим разработчика"))
    else:
        st.sidebar.info(translate("Пользовательский режим"))