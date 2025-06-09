

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15  # increased penalty for not decreasing infected population after 20 days
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + (I_pred_last - I_hat).abs().mean() * last_infected_penalty  
    # term3 = aggregation_func(norm_func(D_hat - D_pred))
    # term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))

    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6)
    return loss