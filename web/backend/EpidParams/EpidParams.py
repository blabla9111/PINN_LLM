import numpy as np
import torch
#https: // www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2020.556689/full
class EpidParams:
    def __init__(self, S, I, R, D, timesteps):
        self.S_data = S
        self.I_data = I
        self.R_data = R
        self.D_data = D
        self.timesteps = timesteps
        self.N = S[0] + I[0] + R[0] + D[0]

    def _diff(self, data, x_left, x_right, delta_x):
        return (data[x_right] - data[x_left]) / (2 * delta_x)

    def beta_t(self, t):
        dSdt = self._diff(self.S_data, t-1, t+1, 1)
        return - dSdt*self.N/self.I_data[t]/self.S_data[t]

    def gamma_t(self, t):
        dRdt = self._diff(self.R_data, t-1, t+1, 1)
        return dRdt/self.I_data[t]

    def delta_t(self, t):
        dDdt = self._diff(self.D_data, t-1, t+1, 1)
        return dDdt/self.I_data[t]

    def Rt(self, t):
        # return self.beta_t(t)/(self.gamma_t(t) + self.delta_t(t))*(self.S_data[t]/self.N)
        # return self.I_data[t]/(self.R_data[t]-self.R_data[t-1]+self.D_data[t]-self.D_data[t-1])
        return torch.sum(self.I_data[t-4:t])/torch.sum(self.I_data[t-8:t-4])

    def R0(self):
        return self.Rt(8)

    def Rt_array(self):
        # почему 8, а не 7?
        Rt_array = np.zeros(len(self.timesteps)-8)
        for i in range(8, len(self.timesteps)):
            Rt_array[i-8] = self.Rt(i)

        return Rt_array
