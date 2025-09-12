import numpy as np

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
        return self.beta_t(t)/(self.gamma_t(t) + self.delta_t(t))*(self.N/self.S_data[t])

    def R0(self):
        return self.Rt(1)

    def Rt_array(self):
        Rt_array = np.zeros(len(self.timesteps)-2)
        for i in range(1, len(self.timesteps)-1):
            Rt_array[i-1] = self.Rt(i)

        return Rt_array
