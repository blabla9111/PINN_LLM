import json
import requests


def send_prompt(prompt_path, LLM_url, file_path_to_save=None):
    with open(prompt_path) as f:
        prompt = json.load(f)

    url = LLM_url
    answer = requests.post(url, json=prompt)

    if file_path_to_save:
        save_answer(file_path_to_save, answer)

    return answer  # Если ОК, то вернется <Response [200]>


def save_answer(file_path_to_save, text):
    with open(file_path_to_save, 'w') as f:
        # Записывается только полученный json (body ответа)
        json.dump(text.text, f, ensure_ascii=False, indent=4)

    return text  # Если ОК, то вернется <Response [200]>


def create_primary_prompt(prompt_file_path, code, comment):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['messages'][1]['content'] = "loss function:\n" + \
        code + "\nExpert comment: \n" + comment + "\n"
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return prompt_file_path


def create_prompt_to_fix_error(prompt_file_path, code, error):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['messages'][1]['content'] = "A previous Python solution code:\n" + \
        code + "\nProblem trace: " + error + ".\n** return only code.**"
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
