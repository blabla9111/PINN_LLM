import subprocess

from loss_dinn_check import TestLossDinnFunction
from lib.loss_update import save
from lib.parser import llm_answer_to_python_code, load_text_to_json, get_loss_func_as_str
from lib.prompt_sender import send_prompt, create_prompt_to_fix_error, create_primary_prompt


def main():

    PROMPT_FILE_PATH = 'prompt_1_1.json'
    PROMPT_FIX_ERROR_FILE_PATH = "prompt_fix_error.json"
    LLM_URL = 'http://localhost:1234/v1/chat/completions'
    ANSWER_FILE_PATH = 'answer_from_LLM_2.json'
    ANSWER_FIX_ERROR_FILE_PATH = 'answer_fix_error_from_LLM_2.json'
    LOSS_FILE_PATH = 'loss_dinn_LLM.py'
    RUN_PINN_COMMAND = ['python', 'PINN.py']
    RUN_TESTER_COMMAND = ['python', 'loss_dinn_check.py']
    EXPERT_COMMENT = "In 20 days the number of infected should start to decrease."


    code = get_loss_func_as_str("loss_dinn_LLM.py")
    create_primary_prompt(PROMPT_FILE_PATH,code, EXPERT_COMMENT )
    send_prompt(PROMPT_FILE_PATH,
                LLM_URL, ANSWER_FILE_PATH)
    print("send prompt")
    json_text = load_text_to_json(ANSWER_FILE_PATH)
    code = llm_answer_to_python_code(json_text)
    print(f'code:\n{code}')
    save(LOSS_FILE_PATH, code)
    print("Run tester")
    output = subprocess.run(RUN_TESTER_COMMAND,
                            capture_output=True, text=True)
    # print(output.stdout)
    t = eval(output.stdout)
    is_correct = t[0]
    error = t[1]
    print(f'tester result = {is_correct}')
    error_counter = 0
    while is_correct == False:
        error_counter+=1
        create_prompt_to_fix_error(PROMPT_FIX_ERROR_FILE_PATH, code, error)
        print(f'error:\n{error}\n\n')
        if error_counter % 3 == 0:
            print("Again send primary prompt")
            send_prompt(PROMPT_FILE_PATH,
                LLM_URL, ANSWER_FILE_PATH)
            json_text = load_text_to_json(ANSWER_FILE_PATH)
        else:
            print("send fix error prompt")
            send_prompt(PROMPT_FIX_ERROR_FILE_PATH,
                LLM_URL, ANSWER_FIX_ERROR_FILE_PATH)
            
            json_text = load_text_to_json(ANSWER_FIX_ERROR_FILE_PATH)
        code = llm_answer_to_python_code(json_text)
        print(f'code:\n{code}')
        save(LOSS_FILE_PATH, code)
        output = subprocess.run(RUN_TESTER_COMMAND,
                            capture_output=True, text=True)
        # print(output.stdout)
        t = eval(output.stdout)
        is_correct = t[0]
        error = t[1]
        # break

    print('RUN PINN')
    output = subprocess.run(RUN_PINN_COMMAND,
                            capture_output=True, text=True)

    print(output.stdout)
    print('done!')


if __name__ == "__main__":
    main()
