import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    # Add penalty to term2 for quantitative characteristics reflecting epidemiological dynamics:
    # During the peak, infected should be about 5% higher than current, so penalize deviation from 1.05 ratio
    peak_index = torch.argmax(I_hat)
    peak_infected_true = I_hat[peak_index]
    peak_infected_pred = I_pred[peak_index]
    peak_ratio_penalty = 10.0 * \
        norm_func(peak_infected_pred / (peak_infected_true + 1e-8) - 1.05)
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + peak_ratio_penalty

    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    # Expert comment: after the peak there should be a sharper decline in the number of infected
    desired_peak_offset = 5  # Offset for the desired peak day
    decline_penalty = 0.1 * \
        torch.mean(norm_func(
            I_pred[peak_index + desired_peak_offset:] - I_hat[peak_index + desired_peak_offset:]))

    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + \
        last_infected_penalty * norm_func(I_pred_last - 0) + decline_penalty
    return loss
