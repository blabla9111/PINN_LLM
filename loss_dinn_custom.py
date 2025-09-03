import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last, x):

    regul = 0.9
    last_infected_penalty = 0.05

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred[:x]))
    term2 = aggregation_func(norm_func(I_hat - I_pred[:x]))
    term3 = aggregation_func(norm_func(D_hat - D_pred[:x]))
    term4 = aggregation_func(norm_func(R_hat - R_pred[:x]))

    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    addition = torch.abs(torch.max(I_pred[x+1:]) - 1)

    # print(term2)
    # print("_"*100)
    # print(term6)
    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + addition
    return loss
