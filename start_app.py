import subprocess

from loss_dinn_check import TestLossDinnFunction
from lib.loss_update import save
from lib.parser import *
from lib.prompt_sender import *
from comment_classificator.match_loss_classification import predict_class_and_sub_class


def main():

    PROMPT_FILE_PATH = 'promts_templates/comment_classifier_1_prompt.json'
    PROMPT_FILE_PATH_2 = 'promts_templates/comment_classifier_2_prompt.json'
    LLM_URL = 'http://localhost:1234/v1/chat/completions'
    ANSWER_FILE_PATH = 'promts_templates/comment_classifier_1_answer.json'
    ANSWER_FILE_PATH_2 = 'promts_templates/comment_classifier_2_answer.json'
    EXPERT_COMMENT = "Mortality in hospitals is 25% with an average world rate of 3-7%."

    RUN_PINN_COMMAND = ['python', 'PINN.py']
    RUN_TESTER_COMMAND = ['python', 'loss_dinn_check.py']
    PROMPT_FIX_ERROR_FILE_PATH = "prompt_fix_error.json"
    ANSWER_FIX_ERROR_FILE_PATH = 'answer_fix_error_from_LLM_2.json'

    top_indices, top_probs = predict_class_and_sub_class(EXPERT_COMMENT)

    # create_comment_class_prompt(PROMPT_FILE_PATH, EXPERT_COMMENT)
    # send_prompt(PROMPT_FILE_PATH,
    #             LLM_URL, ANSWER_FILE_PATH)
    # print("send prompt")
    # json_text = load_text_to_json(ANSWER_FILE_PATH)
    # comment_class = llm_answer_get_comment_class(json_text)
    # comment_class = re.search(r'\d+', comment_class).group()
    print(f'Class:\n{top_indices[0]}  {top_probs[0]}')
    # create_comment_subclass_prompt(
    #     PROMPT_FILE_PATH_2, comment_class, EXPERT_COMMENT)
    # send_prompt(PROMPT_FILE_PATH_2,
    #             LLM_URL, ANSWER_FILE_PATH_2)
    # json_text = load_text_to_json(ANSWER_FILE_PATH_2)
    # comment_subclass = llm_answer_get_comment_class(json_text)
    print(f'Subclass:\n{top_indices[1]}  {top_probs[1]}')

    # comment_class = '4'
    # comment_subclass = '2'

    # PROMPT_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt.json'
    # ANSWER_FILE_PATH = 'promts_templates/get_loss_based_on_recommendation_prompt_answer.json'
    # LOSS_FILE_PATH = 'loss_dinn_LLM.py'
    # LOSS_PRIMARY_FILE_PATH = 'loss_dinn_primary.py'

    # code = get_loss_func_as_str(LOSS_PRIMARY_FILE_PATH)
    # create_get_loss_based_on_recommendation_prompt(PROMPT_FILE_PATH, comment_class, comment_subclass, EXPERT_COMMENT, code)
    # send_prompt(PROMPT_FILE_PATH,
    #             LLM_URL, ANSWER_FILE_PATH)
    # print("send prompt")
    # json_text = load_text_to_json(ANSWER_FILE_PATH)
    # code = llm_answer_to_python_code(json_text)
    # print(f'code:\n{code}')
    # save(LOSS_FILE_PATH, code)
    # print("Run tester")
    # output = subprocess.run(RUN_TESTER_COMMAND,
    #                         capture_output=True, text=True)
    # # print(output.stdout)
    # t = eval(output.stdout)
    # is_correct = t[0]
    # error = t[1]
    # print(f'tester result = {is_correct}')
    # error_counter = 0
    # while is_correct == False:
    #     error_counter += 1
    #     create_prompt_to_fix_error(PROMPT_FIX_ERROR_FILE_PATH, code, error)
    #     print(f'error:\n{error}\n\n')
    #     if error_counter % 3 == 0:
    #         print("Again send primary prompt")
    #         send_prompt(PROMPT_FILE_PATH,
    #                     LLM_URL, ANSWER_FILE_PATH)
    #         json_text = load_text_to_json(ANSWER_FILE_PATH)
    #     else:
    #         print("send fix error prompt")
    #         send_prompt(PROMPT_FIX_ERROR_FILE_PATH,
    #                     LLM_URL, ANSWER_FIX_ERROR_FILE_PATH)

    #         json_text = load_text_to_json(ANSWER_FIX_ERROR_FILE_PATH)
    #     code = llm_answer_to_python_code(json_text)
    #     print(f'code:\n{code}')
    #     save(LOSS_FILE_PATH, code)
    #     output = subprocess.run(RUN_TESTER_COMMAND,
    #                             capture_output=True, text=True)
    #     # print(output.stdout)
    #     t = eval(output.stdout)
    #     is_correct = t[0]
    #     error = t[1]
    #     # break

    # print('RUN PINN')
    # output = subprocess.run(RUN_PINN_COMMAND,
    #                         capture_output=True, text=True)

    # print(output.stdout)
    print('done!')


if __name__ == "__main__":
    main()
