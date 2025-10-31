import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last, train_size):

    S_pred = S_pred[:train_size]
    I_pred = I_pred[:train_size]
    R_pred = R_pred[:train_size]
    D_pred = D_pred[:train_size]

    regul = 0.9
    last_infected_penalty = 0.1

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + \
        last_infected_penalty * norm_func(I_pred_last-0)
    return loss
