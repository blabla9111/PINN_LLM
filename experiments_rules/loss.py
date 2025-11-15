import torch

def data_loss(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, weight = 1.0):

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    return weight * (term1 + term2 + term3 + term4)


def ODE_loss(f1, f2, f3, f4, weight = 1.0):
    aggregation_func = torch.mean
    norm_func = torch.square
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    return weight * (term5 + term6 + term7 + term8)

def initial_boundary_conditions_loss(I_pred_first, I_pred_last, weight = 1.0):
    aggregation_func = torch.mean
    norm_func = torch.square
    
    return aggregation_func(norm_func((I_pred_last - 0)))


def peak_position_loss(I_pred, I_peak_index_expert):
    norm_func = torch.square

    I_pred_peak_index = torch.argmax(I_pred).float()
    
    return norm_func(I_pred_peak_index - I_peak_index_expert)

def peak_height_loss(I_pred, I_peak_height_expert):
    norm_func = torch.square

    return norm_func(torch.max(I_pred).float() - I_peak_height_expert)

def slow_growth_penalty(I_pred, train_size):
    I_pred_peak_index = torch.argmax(I_pred[1:]).item()
    
    # print(f"DEBUG: peak_index={I_pred_peak_index}, train_size={train_size}")
    
    if I_pred_peak_index <= train_size:
        # print("DEBUG: Peak is in training data, returning 0")
        return torch.tensor(0.0, device=I_pred.device)
    
    I_before_peak_after_train_data = I_pred[train_size:I_pred_peak_index+1]
    # print(f"DEBUG: Points between train_size and peak: {len(I_before_peak_after_train_data)}")
    
    if len(I_before_peak_after_train_data) < 2:
        # print("DEBUG: Not enough points for growth calculation, returning 0")
        return torch.tensor(0.0, device=I_pred.device)
    
    relative_growth = (I_before_peak_after_train_data[1:] - I_before_peak_after_train_data[:-1]) / (I_before_peak_after_train_data[:-1] + 1e-8)
    # print(f"DEBUG: Relative growth stats - min: {relative_growth.min().item():.4f}, max: {relative_growth.max().item():.4f}, mean: {relative_growth.mean().item():.4f}")
    
    penalty = torch.exp(-relative_growth * 10)
    result = torch.mean(penalty)
    # print(f"DEBUG: Penalty result: {result.item():.4f}")
    
    return result

def rapid_growth_penalty(I_pred, train_size):
    I_pred_peak_index = torch.argmax(I_pred[1:]).item()
    
    # print(f"DEBUG: peak_index={I_pred_peak_index}, train_size={train_size}")
    
    if I_pred_peak_index <= train_size:
        # print("DEBUG: Peak is in training data, returning 0")
        return torch.tensor(0.0, device=I_pred.device)
    
    I_before_peak_after_train_data = I_pred[train_size:I_pred_peak_index+1]
    # print(f"DEBUG: Points between train_size and peak: {len(I_before_peak_after_train_data)}")
    
    if len(I_before_peak_after_train_data) < 2:
        # print("DEBUG: Not enough points for growth calculation, returning 0")
        return torch.tensor(0.0, device=I_pred.device)
    
    relative_growth = (I_before_peak_after_train_data[1:] - I_before_peak_after_train_data[:-1]) / (I_before_peak_after_train_data[:-1] + 1e-8)
    # print(f"DEBUG: Relative growth stats - min: {relative_growth.min().item():.4f}, max: {relative_growth.max().item():.4f}, mean: {relative_growth.mean().item():.4f}")
    
    penalty = torch.exp(relative_growth * 10)
    result = torch.mean(penalty)
    # print(f"DEBUG: Penalty result: {result.item():.4f}")
    excessive_growth = torch.relu(relative_growth ) # - growth_threshold
    
    # Smoother penalty function
    penalty = torch.log(1.0 + excessive_growth)
    
    return torch.mean(penalty)





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
