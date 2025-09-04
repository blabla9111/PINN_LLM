import json
import requests


def send_prompt(prompt_path, LLM_url, client, file_path_to_save=None):
    with open(prompt_path) as f:
        prompt = json.load(f)

    url = LLM_url
    # answer = requests.post(url, json=prompt)


    # answer = client.chat.completions.create(prompt)
    print(prompt['messages'])
    completion = client.chat.completions.create(
        # Убедитесь что модель поддерживает чат
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": str(prompt["messages"])}],
        max_tokens=500,
        temperature=0.7
    )
    print(completion)

    if file_path_to_save:
        save_answer(file_path_to_save, completion.choices[0].message.content)

    return completion  # Если ОК, то вернется <Response [200]>


def save_answer(file_path_to_save, text):
    with open(file_path_to_save, 'w') as f:
        # Записывается только полученный json (body ответа)
        json.dump(text, f, ensure_ascii=False, indent=4)

    return text  # Если ОК, то вернется <Response [200]>


def create_comment_class_prompt(prompt_file_path, comment):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['messages'][1]['content'] = comment
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return prompt_file_path


def create_comment_subclass_prompt(prompt_file_path, comment_class, comment):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open("rules/comment_subclasses.json", 'r', encoding='utf-8') as f:
        comment_subclasses = json.load(f)
    data['messages'][0]['content'] = "You are a helpful assistant with deep knowledge of epidemiology. Your task is to classify comment. " + comment_subclasses[comment_class] + \
        "\nA comment should be assigned to only one class.\nYou get a comment, but you only have to return its class number."
    data['messages'][1]['content'] = comment
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return prompt_file_path

def create_primary_prompt(prompt_file_path, code, comment):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['messages'][1]['content'] = "loss function:\n" + \
        code + "\nExpert comment: \n" + comment + "\n"
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return prompt_file_path

def create_get_loss_based_on_recommendation_prompt(prompt_file_path, comment_class, comment_subclass, comment, code):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open("rules/recommendations_for_class_subclass.json", 'r', encoding='utf-8') as f:
        recommendations = json.load(f)
    # print(recommendations)
    data['messages'][0][
        'content'] = "You are a machine learning expert specializing in Physics-Informed Neural Networks (PINNs) and epidemiological modeling. Your task is to modify a loss function for a SIRD model to better align with epidemiological dynamics.\nFollow these guidelines strictly:" + recommendations[comment_class]["class_info"] + recommendations[comment_class][comment_subclass] + recommendations[comment_class]["additional_info"] + "\nf1 is the equation for the change in the susceptible population (dS/dt).f2 is the equation for the change in the infected population (dI/dt).f3 is the equation for the change in the recovered population (dR/dt).f4 is the equation for the change in the deceased population (dD/dt). f1 shape: torch.Size([n, m]),f2 shape: torch.Size([n, m]),f3 shape: torch.Size([n]),f4 shape: torch.Size([n]). S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4 are tensors of the forms tensor([float_1, float_2, ... , float_n]). I_pred_last is a 0-d tensor. loss should be tensor(float_num, dtype=torch.float64, grad_fn=<AddBackward0>).\nStart your answer with 'def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):'. Return only the code."
    data['messages'][1]['content'] = "loss function:\n" + \
        code + "\nExpert comment: \n" + comment
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def create_prompt_to_fix_error(prompt_file_path, code, error):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['messages'][1]['content'] = "A previous Python solution code:\n" + \
        code + "\nProblem trace: " + error + ".\n** return only code.**"
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
