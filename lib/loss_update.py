from loss_dinn_check import TestLossDinnFunction

def save(py_file_path, code, add_imports=True, history_file_path = 'losses.py'):

    code = "\n\nimport torch\n\n\n" + code  # костыль
    with open(py_file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # нужна проверка на корректность лосса!
    save_to_history(history_file_path, code)
    return py_file_path

def save_to_history(py_file_path, code, add_imports=True):
    with open(py_file_path, 'a', encoding='utf-8') as f:
        f.write(code)
        
    return py_file_path

