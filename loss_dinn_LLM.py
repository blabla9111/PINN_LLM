import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):


    regul = 0.8
    last_infected_penalty = 0.05
    decline_penalty = 0.1
    peak_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    # Find the peak of predicted infections
    I_pred_max = torch.max(I_pred)
    I_hat_max = torch.max(I_hat)

    # Penalty for peak discrepancy (expert expects 10% higher peak)
    expected_peak = I_hat_max * 1.1
    peak_discrepancy = norm_func(I_pred_max - expected_peak)

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + \
        decline_penalty * aggregation_func(torch.relu(f2)) + \
        peak_penalty * peak_discrepancy
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 +
                                                                    term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss
