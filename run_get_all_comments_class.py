import subprocess
from loss_dinn_check import TestLossDinnFunction
from lib.loss_update import save
from lib.parser import *
from lib.prompt_sender import *


def process_comment(comment, prompt_template_path, answer_path, llm_url):
    """Обрабатывает один комментарий и возвращает его класс"""
    create_comment_class_prompt(prompt_template_path, comment)
    send_prompt(prompt_template_path, llm_url, answer_path)
    json_text = load_text_to_json(answer_path)
    return llm_answer_get_comment_class(json_text)


def main():
    # Конфигурационные параметры
    PROMPT_FILE_PATH = 'promts_templates/comment_classifier_1_prompt.json'
    PROMPT_FILE_PATH_2 = 'promts_templates/comment_classifier_2_prompt.json'
    LLM_URL = 'http://localhost:1234/v1/chat/completions'
    ANSWER_FILE_PATH = 'promts_templates/comment_classifier_1_answer.json'
    ANSWER_FILE_PATH_2 = 'promts_templates/comment_classifier_2_answer.json'
    OUTPUT_FILE = 'all_classification_results.txt'
    COMMENTS_FILE = 'comments_examples.txt'

    # Чтение комментариев из файла
    with open(COMMENTS_FILE, 'r', encoding='utf-8') as f:
        comments = [line.strip() for line in f if line.strip()]

    results = []

    # Обработка каждого комментария
    for i, comment in enumerate(comments, 1):
        print(f"Processing comment {i}/{len(comments)}: {comment[:50]}...")
        try:
            class_result = process_comment(
                comment,
                PROMPT_FILE_PATH,
                ANSWER_FILE_PATH,
                LLM_URL
            )
            results.append(
                f"Comment: {comment}\nClass: {class_result}\n{'='*50}\n")
            print(f"Result: Class {class_result}")
        except Exception as e:
            results.append(f"Comment: {comment}\nError: {str(e)}\n{'='*50}\n")
            print(f"Error processing comment: {e}")

    # Запись всех результатов в файл
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(results)

    print(f"Done! All results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
