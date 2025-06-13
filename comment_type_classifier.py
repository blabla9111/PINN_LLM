import subprocess

from loss_dinn_check import TestLossDinnFunction
from lib.loss_update import save
from lib.parser import *
from lib.prompt_sender import *


def main():

    PROMPT_FILE_PATH = 'promts_templates/comment_classifier_1_prompt.json'
    PROMPT_FILE_PATH_2 = 'promts_templates/comment_classifier_2_prompt.json'
    LLM_URL = 'http://localhost:1234/v1/chat/completions'
    ANSWER_FILE_PATH = 'promts_templates/comment_classifier_1_answer.json'
    ANSWER_FILE_PATH_2 = 'promts_templates/comment_classifier_2_answer.json'
    EXPERT_COMMENT = "on the 20th day the vaccination started - this is not visible on the graphs"


    create_comment_class_prompt(PROMPT_FILE_PATH,EXPERT_COMMENT )
    send_prompt(PROMPT_FILE_PATH,
                LLM_URL, ANSWER_FILE_PATH)
    print("send prompt")
    json_text = load_text_to_json(ANSWER_FILE_PATH)
    comment_class = llm_answer_get_comment_class(json_text)
    print(f'Class:\n{comment_class}')
    create_comment_subclass_prompt(
        PROMPT_FILE_PATH_2, comment_class, EXPERT_COMMENT)
    send_prompt(PROMPT_FILE_PATH_2,
                LLM_URL, ANSWER_FILE_PATH_2)
    json_text = load_text_to_json(ANSWER_FILE_PATH_2)
    comment_subclass = llm_answer_get_comment_class(json_text)
    print(f'Subclass:\n{comment_subclass}')


    print('done!')


if __name__ == "__main__":
    main()
