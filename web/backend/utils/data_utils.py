import pandas as pd

def get_data_for_model(filepath: str):
    """Получить данные для модели из CSV файла"""
    x = 180  # сколько дней для обучения

    covid_cases = pd.read_csv(filepath)
    susceptible = []
    infected = []
    dead = []
    recovered = []
    timesteps = []

    d1 = covid_cases["S"]
    d2 = covid_cases["I"]
    d3 = covid_cases["D"]
    d4 = covid_cases["R"]
    d5 = covid_cases["t"]

    for item in range(len(d5)):
        if item % 1 == 0:
            susceptible.append(d1[item])
            infected.append(d2[item])
            dead.append(d3[item])
            recovered.append(d4[item])
            timesteps.append(d5[item])

    return timesteps, susceptible, infected, dead, recovered, x