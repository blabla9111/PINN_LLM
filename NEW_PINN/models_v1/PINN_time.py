import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class NetStates(nn.Module):
    '''нейросеть для предсказывания SIRD'''

    def __init__(self, layers=[1, 200, 100, 4]):
        super().__init__()

        # TODO нужно подумать насчет кол-ва нейронов
        # почему не степень 2?
        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 100)
        self.out = nn.Linear(100, 4)

        # Инициализация Xavier
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, t):
        x = torch.tanh(self.fc1(t))
        x = torch.tanh(self.fc2(x))
        return self.out(x)


class NetParameter(nn.Module):
    '''Нейросеть для параметра как функции от времени'''

    def __init__(self, layers=[1, 50, 50, 1]):
        super().__init__()

        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, t):
        x = torch.tanh(self.fc1(t))
        x = torch.tanh(self.fc2(x))
        return self.out(x)


class Nature_PINN(nn.Module):
    def __init__(self, t, S_data, I_data, R_data, D_data, device, train_size, population_n):
        super(Nature_PINN, self).__init__()

        self.device = device
        self.N = population_n
        self.train_size = train_size

        # Данные
        self.t = torch.tensor(t, requires_grad=True,
                              device=self.device).float()
        self.t_batch = self.t.reshape(-1, 1)

        self.S = torch.tensor(S_data, device=device).float().reshape(-1, 1)
        self.I = torch.tensor(I_data, device=device).float().reshape(-1, 1)
        self.R = torch.tensor(R_data, device=device).float().reshape(-1, 1)
        self.D = torch.tensor(D_data, device=device).float().reshape(-1, 1)

        # Границы для масштабирования времени
        self.lb = torch.tensor(0, device=device).float()
        self.ub = torch.tensor(torch.max(self.t), device=device).float()

        # Масштабирование времение в [-1, 1] (как в статье)
        # self.t_scaled = 2.0 * (self.t_batch - self.lb) / (self.ub - self.lb) - 1.0

        # Нормализация данных
        self.S_max = self.S[:train_size].max()
        self.I_max = self.I[:train_size].max()
        self.D_max = self.D[:train_size].max()
        self.R_max = self.R[:train_size].max()
        self.S_min = self.S[:train_size].min()
        self.I_min = self.I[:train_size].min()
        self.D_min = self.D[:train_size].min()
        self.R_min = self.R[:train_size].min()

        self.S_hat = (self.S - self.S_min) / (self.S_max - self.S_min)
        self.I_hat = (self.I - self.I_min) / (self.I_max - self.I_min)
        self.D_hat = (self.D - self.D_min) / (self.D_max - self.D_min)
        self.R_hat = (self.R - self.R_min) / (self.R_max - self.R_min)

        # Инициализация сетей
        # Сеть для состояний
        self.net_states = NetStates().to(self.device)

        # Отдельные сети для параметров (как в статье)
        self.net_beta = NetParameter().to(self.device)
        self.net_gamma = NetParameter().to(self.device)
        self.net_mu = NetParameter().to(self.device)

        # Все обучаемые параметры
        self.params = (list(self.net_states.parameters()) +
                       list(self.net_beta.parameters()) +
                       list(self.net_gamma.parameters()) +
                       list(self.net_mu.parameters()))

        # Для логирования
        self.losses = []

    @property
    def beta(self):
        '''β(t) в диапазоне [0, 1]'''
        return torch.sigmoid(self.net_beta(self.t_scaled))

    @property
    def gamma(self):
        '''γ(t) в диапазоне [0, 0.5] -- предполагается, что меньше 2-х дней люди не болеют'''
        return 0.5 * torch.sigmoid(self.net_gamma(self.t_scaled))

    @property
    def mu(self):
        '''μ(t) в диапазоне [0, 0.1] -- считаем, что летальность выше 10% маловероятна для гриппа'''
        return 0.1 * torch.sigmoid(self.net_mu(self.t_scaled))

    def net_f(self):
        '''ВЫчисление невязок'''
        t_batch = self.t_batch.clone().detach().requires_grad_(True)

        t_scaled = 2.0 * (t_batch - self.lb) / (self.ub - self.lb) - 1.0

        # Получение предсказания (нормализованные)
        states_hat = self.net_states(t_scaled)
        S_hat, I_hat, R_hat, D_hat = states_hat[:, 0:1], states_hat[:,
                                                                    1:2], states_hat[:, 2:3], states_hat[:, 3:4]

        # Денормализация
        S = self.S_min + (self.S_max - self.S_min) * S_hat
        I = self.I_min + (self.I_max - self.I_min) * I_hat
        R = self.R_min + (self.R_max - self.R_min) * R_hat
        D = self.D_min + (self.D_max - self.D_min) * D_hat

        # Параметры эпидемии
        # beta =  self.beta
        # gamma = self.gamma
        # mu = self.mu

        # Производные нормализованных переменных
        S_hat_t = torch.autograd.grad(S_hat.sum(), t_batch,
                                      create_graph=True)[0]
        I_hat_t = torch.autograd.grad(I_hat.sum(), t_batch,
                                      create_graph=True)[0]
        R_hat_t = torch.autograd.grad(R_hat.sum(), t_batch,
                                      create_graph=True)[0]
        D_hat_t = torch.autograd.grad(D_hat.sum(), t_batch,
                                      create_graph=True)[0]

        # Восстановление оригинальных производных (в тетради\miro есть вывод формулы)
        S_t = (self.S_max - self.S_min) * S_hat_t
        I_t = (self.I_max - self.I_min) * I_hat_t
        R_t = (self.R_max - self.R_min) * R_hat_t
        D_t = (self.D_max - self.D_min) * D_hat_t

        beta = torch.sigmoid(self.net_beta(t_scaled)) + 0.01
        gamma = 0.5 * torch.sigmoid(self.net_gamma(t_scaled)) + 0.03
        mu = 0.1 * torch.sigmoid(self.net_mu(t_scaled)) + 0.0001

        # Уравнения SIRD
        dS = - beta * I * S / self.N
        dI = beta * I * S / self.N - gamma * I - mu * I
        dR = gamma * I
        dD = mu * I

        # Невязки в реальных масштабах
        f_S = S_t - dS
        f_I = I_t - dI
        f_R = R_t - dR
        f_D = D_t - dD

        # Conservation (как в статье)
        # TODO пока без этого попробую
        # можно нормализовать для loss
        f_con = (S + I + R + D - self.N) / self.N

        # для ошибки по данным возьму нормализованные значения
        return f_S, f_I, f_R, f_D, S_hat, I_hat, R_hat, D_hat, beta, gamma, mu

    def loss_function(self):
        '''Вычисление функции потерь'''
        f_S, f_I, f_R, f_D, S_pred, I_pred, R_pred, D_pred, beta, gamma, mu = self.net_f()

        # Индексы для обучающей выборки
        idx = slice(self.train_size)

        # Ошибка на данных
        loss_data = (1.0 * torch.mean((S_pred[idx] - self.S_hat[idx])**2) +
                     1.0 * torch.mean((I_pred[idx] - self.I_hat[idx])**2) +
                     1.0 * torch.mean((R_pred[idx] - self.R_hat[idx])**2) +
                     1.0 * torch.mean((D_pred[idx] - self.D_hat[idx])**2))

        # Ошибка на начальных условиях
        # TODO пока отключу, это и так есть в невязке по данным
        loss_ic = (torch.mean((S_pred[0] - self.S_hat[0])**2) +
                   torch.mean((I_pred[0] - self.I_hat[0])**2) +
                   torch.mean((R_pred[0] - self.R_hat[0])**2) +
                   torch.mean((D_pred[0] - self.D_hat[0])**2))

        # Невязка по уравнениям SIRD
        loss_phys = (torch.mean(f_S**2) +
                     torch.mean(f_I**2) +
                     torch.mean(f_R**2) +
                     torch.mean(f_D**2))

        # Регуляризация параметров (для гладкости)
        # TODO пока без этого, если значения будут прыгать, то включу
        # beta, gamma, mu = self.beta, self.gamma, self.mu
        loss_reg = (torch.mean((beta[1:] - beta[:-1])**2) +
                    torch.mean((gamma[1:] - gamma[:-1])**2) +
                    torch.mean((mu[1:] - mu[:-1])**2)) * 0.001

        # Итоговая потеря
        total_loss = (1.0 * loss_data +
                      1.0 * loss_phys)

        return total_loss, loss_data, loss_phys

    def train(self, n_epoch=20000):
        '''Обучение модели'''
        print('\n' + '='*60)
        print('НАЧАЛО ОБУЧЕНИЯ')
        print('='*60)

        optimizer = optim.Adam(self.params, lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000
        )

        for epoch in range(n_epoch):
            optimizer.zero_grad()

            total_loss, loss_data, loss_phys = self.loss_function()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss)

            self.losses.append(total_loss.item())

            if epoch % 1000 == 0:
                print(f'\nЭпоха {epoch:5d}:')
                print(f'  Total Loss: {total_loss.item():.2e}')
                print(f'  Data Loss: {loss_data.item():.2e}')
                print(f'  Physics Loss: {loss_phys.item():.2e}')

                # Статистика параметров
                with torch.no_grad():

                    t_batch = self.t_batch
                    t_scaled = 2.0 * (t_batch - self.lb) / \
                        (self.ub - self.lb) - 1.0

                    beta = torch.sigmoid(self.net_beta(t_scaled)) + 0.01
                    gamma = 0.5 * \
                        torch.sigmoid(self.net_gamma(t_scaled)) + 0.03
                    mu = 0.1 * torch.sigmoid(self.net_mu(t_scaled)) + 0.0001

                    beta_mean = beta.mean().item()
                    gamma_mean = gamma.mean().item()
                    mu_mean = mu.mean().item()

                    print(
                        f'  β mean: {beta_mean:.4f} [min: {beta.min().item():.4f}, max: {beta.max().item():.4f}]')
                    print(
                        f'  γ mean: {gamma_mean:.4f} [min: {gamma.min().item():.4f}, max: {gamma.max().item():.4f}]')
                    print(
                        f'  μ mean: {mu_mean:.4f} [min: {mu.min().item():.4f}, max: {mu.max().item():.4f}]')

    def predict(self, t_new=None):
        '''Предсказания'''
        if t_new is None:
            t_tensor = self.t_batch
        else:
            t_tensor = torch.tensor(
                t_new, device=self.device).float().reshape(-1, 1)
        t_scaled = 2.0 * (t_tensor - self.lb) / (self.ub - self.lb) - 1.0

        with torch.no_grad():
            states_hat = self.net_states(t_scaled)
            S_hat, I_hat, R_hat, D_hat = states_hat[:,
                                                    0], states_hat[:, 1], states_hat[:, 2], states_hat[:, 3]

            # Денормализация
            S = self.S_min + (self.S_max - self.S_min) * S_hat
            I = self.I_min + (self.I_max - self.I_min) * I_hat
            R = self.R_min + (self.R_max - self.R_min) * R_hat
            D = self.D_min + (self.D_max - self.D_min) * D_hat

            # Параметры (t_scaled могут быть новыми, поэтому не property вызов)
            beta = torch.sigmoid(self.net_beta(t_scaled)).flatten() + 0.01
            gamma = 0.5 * \
                torch.sigmoid(self.net_gamma(t_scaled)).flatten() + 0.03
            mu = 0.1 * torch.sigmoid(self.net_mu(t_scaled)).flatten() + 0.0001

        return {
            'S': S.cpu().detach().numpy(),
            'I': I.cpu().detach().numpy(),
            'R': R.cpu().detach().numpy(),
            'D': D.cpu().detach().numpy(),
            'beta': beta.cpu().detach().numpy(),
            'gamma': gamma.cpu().detach().numpy(),
            'mu': mu.cpu().detach().numpy(),
            't': t_tensor.cpu().detach().numpy().flatten()
        }
