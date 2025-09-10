import torch
import torch.nn as nn
import torch.nn.functional as F
# from loss_dinn_LLM import loss_dinn

class DINN(nn.Module):
    def __init__(self, t, S_data, I_data, D_data, R_data):
        super(DINN, self).__init__()
        self.N = 6e6
        self.t = torch.tensor(t, requires_grad=True)
        self.t_float = self.t.float()
        self.t_batch = torch.reshape(self.t_float, (len(self.t), 1))
        self.S = torch.tensor(S_data)
        self.I = torch.tensor(I_data)
        self.D = torch.tensor(D_data)
        self.R = torch.tensor(R_data)

        self.losses = []

        self.beta_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.gamma_tilda = torch.nn.Parameter(
            torch.rand(1, requires_grad=True))

        self.S_max = max(self.S)
        self.I_max = max(self.I)
        self.D_max = max(self.D)
        self.R_max = max(self.R)
        self.S_min = min(self.S)
        self.I_min = min(self.I)
        self.D_min = min(self.D)
        self.R_min = min(self.R)

        self.S_hat = (self.S - self.S_min) / (self.S_max - self.S_min)
        self.I_hat = (self.I - self.I_min) / (self.I_max - self.I_min)
        self.D_hat = (self.D - self.D_min) / (self.D_max - self.D_min)
        self.R_hat = (self.R - self.R_min) / (self.R_max - self.R_min)

        self.m1 = torch.zeros((len(self.t), 4))
        self.m1[:, 0] = 1
        self.m2 = torch.zeros((len(self.t), 4))
        self.m2[:, 1] = 1
        self.m3 = torch.zeros((len(self.t), 4))
        self.m3[:, 2] = 1
        self.m4 = torch.zeros((len(self.t), 4))
        self.m4[:, 3] = 1

        self.net_sidr = self.Net_sidr()
        self.params = list(self.net_sidr.parameters())
        self.params.extend([self.beta_tilda, self.gamma_tilda])

    @property
    def beta(self):
        return torch.tanh(self.beta_tilda)

    @property
    def gamma(self):
        return torch.tanh(self.gamma_tilda)

    class Net_sidr(nn.Module):
        def __init__(self):
            super(DINN.Net_sidr, self).__init__()
            self.fc1 = nn.Linear(1, 200)
            self.fc2 = nn.Linear(200, 100)
            self.out = nn.Linear(100, 4)
            self.out_alpha = nn.Linear(100, 1)

        def forward(self, t_batch):
            x = F.relu(self.fc1(t_batch))
            x = F.tanh(self.fc2(x))
            sidr = self.out(x)
            alpha = self.out_alpha(x)
            return sidr, alpha

    def net_f(self, t_batch):
        sidr_hat, alpha_hat = self.net_sidr(t_batch)

        S_hat, I_hat, D_hat, R_hat = sidr_hat[:,
                                              0], sidr_hat[:, 1], sidr_hat[:, 2], sidr_hat[:, 3]

        # S_t
        sidr_hat.backward(self.m1, retain_graph=True)
        S_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # I_t
        sidr_hat.backward(self.m2, retain_graph=True)
        I_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # D_t
        sidr_hat.backward(self.m3, retain_graph=True)
        D_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # R_t
        sidr_hat.backward(self.m4, retain_graph=True)
        R_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # Unnormalize
        S = self.S_min + (self.S_max - self.S_min) * S_hat
        I = self.I_min + (self.I_max - self.I_min) * I_hat
        D = self.D_min + (self.D_max - self.D_min) * D_hat
        R = self.R_min + (self.R_max - self.R_min) * R_hat

        f1_hat = S_hat_t - (-(alpha_hat / self.N) * S * I) / \
            (self.S_max - self.S_min)
        f2_hat = I_hat_t - ((alpha_hat / self.N) * S * I - self.beta.squeeze()
                            * I - self.gamma * I) / (self.I_max - self.I_min)
        f3_hat = D_hat_t - (self.gamma * I) / (self.D_max - self.D_min)
        f4_hat = R_hat_t - (self.beta.squeeze() * I) / \
            (self.R_max - self.R_min)

        return f1_hat, f2_hat, f3_hat, f4_hat, S_hat, I_hat, D_hat, R_hat, alpha_hat

    # def train(self, n_epochs, regul):
    #     # Train
    #     print('\nStarting training...\n')

    #     for epoch in range(n_epochs):
    #         S_pred_list = []
    #         I_pred_list = []
    #         D_pred_list = []
    #         R_pred_list = []
    #         alpha_pred_list = []

    #         f1, f2, f3, f4, S_pred, I_pred, D_pred, R_pred, alpha_pred = self.net_f(
    #             self.t_batch)
    #         self.optimizer.zero_grad()
    #         x = 180 # костыль !!!!!!
    #         S_pred_list.append(self.S_min + (self.S_max - self.S_min) * S_pred)
    #         I_pred_list.append(self.I_min + (self.I_max - self.I_min) * I_pred)
    #         D_pred_list.append(self.D_min + (self.D_max - self.D_min) * D_pred)
    #         R_pred_list.append(self.R_min + (self.R_max - self.R_min) * R_pred)
    #         alpha_pred_list.append(alpha_pred)
    #         loss = loss_dinn(self.S_hat[:x], S_pred[:x],
    #                          self.I_hat[:x], I_pred[:x],
    #                          self.D_hat[:x], D_pred[:x],
    #                          self.R_hat[:x], R_pred[:x],
    #                          f1[:x],
    #                          f2[:x],
    #                          f3[:x],
    #                          f4[:x], I_pred[-1])
    #         # print("!!!!!!!!!!!!")
    #         loss.backward()
    #         self.optimizer.step()
    #         self.scheduler.step()

    #         self.losses.append(loss.item())

    #         if epoch % 1000 == 0:
    #             print('\nEpoch ', epoch)

    #         # Loss + model parameters update
    #         if epoch % 4000 == 0:
    #             print('Loss is: ', loss)
    #             print('Epoch: ', epoch)
    #             print('dinn.beta', self.beta)
    #             print('dinn.gamma', self.gamma)
    #             print(alpha_pred.shape)

    #     return S_pred_list, I_pred_list, D_pred_list, R_pred_list, alpha_pred_list

    def predict(self, t_values=None):
        """Получить прогноз модели для заданных временных точек"""
        if t_values is None:
            t_values = self.t_float

        t_batch = torch.reshape(t_values, (len(t_values), 1))

        with torch.no_grad():
            sidr_hat, alpha_hat = self.net_sidr(t_batch)
            S_hat, I_hat, D_hat, R_hat = sidr_hat[:,
                                                  0], sidr_hat[:, 1], sidr_hat[:, 2], sidr_hat[:, 3]

            # Денормализация
            S_pred = self.S_min + (self.S_max - self.S_min) * S_hat
            I_pred = self.I_min + (self.I_max - self.I_min) * I_hat
            D_pred = self.D_min + (self.D_max - self.D_min) * D_hat
            R_pred = self.R_min + (self.R_max - self.R_min) * R_hat

        return S_pred, I_pred, D_pred, R_pred, alpha_hat
