import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.005  # Reduced penalty for the last infected population
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) * 0.5  # Reduced penalty for the infection rate forecast
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 10) else torch.mean # reflect quarantine effect after day 10
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * norm_func(f1+f2+f3+f4) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for the difference between predicted and actual infected cases after day 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_day_10 = 0.2
    
    loss_before_day_10 = regul * (term1 + term2 + term3 + term4)
    
    # Calculate the penalty for the difference between predicted and actual infected cases after day 10
    term5 = aggregation_func(norm_func(I_hat - I_pred))
    last_infected_penalty_after_day_10_term = aggregation_func(norm_func(I_pred_last-0))

    loss_after_day_10 = (1 - regul) * (term5 + last_infected_penalty_after_day_10_term)
    
    # Combine the two penalties
    loss = loss_before_day_10 + loss_after_day_10 + quarantine_penalty * norm_func((I_hat - I_pred).abs()) + last_infected_penalty * norm_func(I_pred_last-0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for the difference between predicted and actual infected population after day 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_day_10 = 0.2

    loss_before_day_10 = regul * (term1 + term2 + term3 + term4)
    loss_after_day_10 = (1 - regul) * (aggregation_func(norm_func(f1)) + aggregation_func(norm_func(f2)) + 
                                      aggregation_func(norm_func(f3)) + aggregation_func(norm_func(f4)))
    
    # Combine the two parts of the loss function
    loss = loss_before_day_10 + loss_after_day_10 + last_infected_penalty * norm_func(I_pred_last-0) + \
           quarantine_penalty * norm_func((I_hat - I_pred).gt(10)) + last_infected_penalty_after_day_10 * norm_func((I_hat - I_pred).gt(10))
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 10) else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (10 > 9) else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred_last > 10) else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if day > 15 else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased penalty for late outbreak
    aggregation_func = torch.max # Use max instead of mean to prioritize earlier outbreaks
    norm_func = torch.abs # Use absolute difference instead of squared difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # Changed to max for early outbreak
    norm_func = torch.abs # Changed to absolute difference for early outbreak
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max
    norm_func = torch.abs
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max  # Changed to maximize the difference between predictions and actuals
    norm_func = torch.abs  # Changed to absolute difference for easier interpretation
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    loss = regul * (term2) + (1 - regul) * (norm_func(f2)) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.5
    aggregation_func = torch.max
    norm_func = torch.abs
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    loss = regul * (term2) + (1 - regul) * (norm_func(f2)) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.9
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    #term3 = aggregation_func(norm_func(D_hat - D_pred))
    #term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    #term7 = aggregation_func(norm_func(f3))
    #term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.95
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    #term3 = aggregation_func(norm_func(D_hat - D_pred)) 
    #term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    #term7 = aggregation_func(norm_func(f3))
    #term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # Changed to max instead of mean
    norm_func = torch.abs # Changed to absolute difference instead of square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # Removed the term
    term4 = 0 # Removed the term
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.9
    last_infected_penalty = 0.02
    aggregation_func = torch.max
    norm_func = torch.abs
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.95
    last_infected_penalty = 0.05
    aggregation_func = torch.max
    norm_func = torch.abs
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + last_infected_penalty * norm_func(I_pred_last-0)
    #term3, term4, term5, term6, term7, term8 are removed 
    loss = regul * term1 + (1 - regul) * term2
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased the penalty to reflect the late outbreak
    aggregation_func = torch.max # Changed to max to prioritize recent predictions
    norm_func = torch.abs # Changed to absolute difference to avoid squaring large differences
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + last_infected_penalty * norm_func(I_hat - I_pred) # Added penalty for late infected population
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased the penalty to account for delayed outbreak
    aggregation_func = torch.max # Changed to max to prioritize recent predictions
    norm_func = torch.abs # Changed to absolute difference to avoid squaring errors
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + last_infected_penalty * norm_func(I_pred_last-0) # Added penalty directly to term2
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term3 + term4) + (1 - regul) * (term2)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased penalty for late outbreak
    aggregation_func = torch.max # Changed to max function to prioritize recent data
    norm_func = torch.abs # Changed to absolute difference to avoid negative values
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + last_infected_penalty * norm_func(I_pred_last-0) # Added penalty for late outbreak
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # increased penalty for late outbreak
    aggregation_func = torch.max # use max instead of mean to prioritize earlier predictions
    norm_func = torch.abs # use absolute difference instead of squared difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred)) # removed this term to focus on infection and recovery
    term4 = aggregation_func(norm_func(R_hat - R_pred)) # removed this term to focus on infection and recovery
    loss = regul * (term1 + term2) + (1 - regul) * (term3 + term4) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased penalty for late outbreak
    aggregation_func = torch.max # Changed to maximize the difference between predicted and actual values
    norm_func = torch.abs # Changed to absolute difference instead of squared difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased penalty for late outbreak
    aggregation_func = torch.max # Changed to max function to prioritize latest data
    norm_func = torch.abs # Changed to absolute difference to reduce effect of early data
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + last_infected_penalty * norm_func(I_pred_last-0) # Added penalty for late outbreak
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term3 + term4) + (1 - regul) * (term2)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max for early outbreak
    norm_func = torch.abs # changed from square to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return lossdef loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return lossdef loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max 
    norm_func = torch.abs 
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max 
    norm_func = torch.abs 
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max 
    norm_func = torch.abs 
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max 
    norm_func = torch.abs 
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # changed to max to align with expert expectations
    norm_func = torch.abs # changed to absolute difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return los

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max 
    norm_func = torch.abs 
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = 0 # removed term as it's not used in the function
    term4 = 0 # removed term as it's not used in the function
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased the penalty for late infected population
    aggregation_func = torch.max # Changed to max function to prioritize latest data
    norm_func = torch.abs # Changed to absolute difference to avoid negative values
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 
    aggregation_func = torch.max 
    norm_func = torch.abs 
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define and calculate missing terms
    term5 = aggregation_func(norm_func(f1 - f2))  # Assuming f1 and f2 are the predicted and actual values for the first variable
    term6 = aggregation_func(norm_func(f3 - f4))  # Assuming f3 and f4 are the predicted and actual values for the second variable
    term7 = aggregation_func(norm_func(I_pred_last - I_hat))
    term8 = aggregation_func(norm_func(D_pred - D_hat))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # Changed to maximize the difference between predicted and actual values
    norm_func = torch.abs # Changed to absolute difference instead of squared difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) * last_infected_penalty # Increased penalty for late outbreak
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term3 + term4) + (1 - regul) * (term2) 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # Changed to max for early outbreak detection
    norm_func = torch.abs # Changed to absolute difference for robustness
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + last_infected_penalty * norm_func(I_pred_last-0) # Added penalty directly to term2
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term3 + term4) + (1 - regul) * (term2)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max
    norm_func = torch.abs
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    #term3 = aggregation_func(norm_func(D_hat - D_pred)) # Removed term3
    #term4 = aggregation_func(norm_func(R_hat - R_pred))  # Removed term4
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased penalty for late outbreak
    aggregation_func = torch.max # Use max function to prioritize earlier predictions
    norm_func = torch.abs # Use absolute difference instead of squared difference
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased penalty for late outbreak
    aggregation_func = torch.max # Use max function to prioritize earlier predictions
    norm_func = torch.abs # Use absolute difference instead of squared difference

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define missing terms
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last))  # Assuming this is the correct term
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.9
    last_infected_penalty = 0.5
    aggregation_func = torch.max
    norm_func = torch.abs
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    #term3 = aggregation_func(norm_func(D_hat - D_pred))
    #term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    #term7 = aggregation_func(norm_func(f3))
    #term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2)  + (1 - regul) * (term5 + term6) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.001 # Reduced the penalty for late outbreak
    aggregation_func = torch.max # Changed to max to prioritize earlier predictions
    norm_func = torch.abs # Changed to absolute difference to avoid squared differences
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.001 # Reduced the penalty for late outbreak
    aggregation_func = torch.max # Changed to max to prioritize earlier predictions
    norm_func = torch.abs # Changed to absolute difference to avoid squared differences

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define the missing terms
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last))  # Assuming this is the correct term
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max # Changed to maximize the infected population
    norm_func = torch.norm
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) * regul # Multiply by regul to weigh it properly
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term3 + term4) + (1 - regul) * term2  + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15
    aggregation_func = torch.max
    norm_func = torch.abs
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15
    aggregation_func = torch.max
    norm_func = torch.abs
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define missing terms
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last))  # Assuming this is the correct term
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 10) else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 10) else torch.mean
    norm_func = torch.square
    
    # Fix the error by adding term5, term6, term7 and term8 before calculating loss
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Calculate the remaining terms
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last))  # Use I_pred_last instead of I_pred
    term8 = aggregation_func(norm_func(D_hat - D_pred))      # This line was commented out
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 10) else torch.mean
    norm_func = torch.square
    
    # Fix the error by adding term5, term6, term7 and term8 before calculating loss
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))  # This line was commented out
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Calculate the remaining terms
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last))  # Use I_pred_last instead of I_pred
    term8 = aggregation_func(norm_func(D_hat - D_pred))      # This line was commented out
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.2
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(f1) + norm_func(f2) + norm_func(f3) + norm_func(f4))
    
    # Calculate the difference in infected population for days after 10
    I_pred_diff = torch.where(I_pred_last > 0, I_hat - I_pred, torch.zeros_like(I_hat))
    
    loss_after_10 = regul * (term1[:10] + term2[:10] + term3[:10] + term4[:10]) + (1 - regul) * (norm_func(f1[:10]) + norm_func(f2[:10]) + norm_func(f3[:10]) + norm_func(f4[:10])) + last_infected_penalty_after_10 * aggregation_func(norm_func(I_pred_diff))
    
    loss = loss_before_10 + loss_after_10
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.2
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(f1) + norm_func(f2) + norm_func(f3) + norm_func(f4))
    
    # Calculate the difference in infected population for days after 10
    I_pred_diff = torch.where(I_pred_last > 0, I_hat - I_pred, torch.zeros_like(I_hat))
    
    loss_after_10 = regul * (term1[:10] if len(term1) > 10 else term1) + \
                    regul * (term2[:10] if len(term2) > 10 else term2) + \
                    regul * (term3[:10] if len(term3) > 10 else term3) + \
                    regul * (term4[:10] if len(term4) > 10 else term4) + \
                    (1 - regul) * (norm_func(f1[:10]) if len(f1) > 10 else norm_func(f1)) + \
                    (1 - regul) * (norm_func(f2[:10]) if len(f2) > 10 else norm_func(f2)) + \
                    (1 - regul) * (norm_func(f3[:10]) if len(f3) > 10 else norm_func(f3)) + \
                    (1 - regul) * (norm_func(f4[:10]) if len(f4) > 10 else norm_func(f4)) + \
                    last_infected_penalty_after_10 * aggregation_func(norm_func(I_pred_diff))
    
    loss = loss_before_10 + loss_after_10
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 100) else torch.mean # Introduce a threshold to switch from mean to max after day 10
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(f1) + norm_func(f2) + norm_func(f3) + norm_func(f4)) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 100) else torch.mean # Introduce a threshold to switch from mean to max after day 10
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(f1) + norm_func(f2) + norm_func(f3) + norm_func(f4)) + last_infected_penalty * norm_func(torch.tensor([I_pred_last-0]))
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 100) else torch.mean # Introduce a threshold to switch from mean to max after day 10
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(f1) + norm_func(f2) + norm_func(f3) + norm_func(f4)) + last_infected_penalty * norm_func(torch.tensor([I_pred_last-0]))
    
    # Convert tensor to scalar before applying the penalty
    loss = loss + last_infected_penalty * norm_func(I_pred_last - 0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 10).any() else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred > 10).any() else torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5, 6, 7 and 8
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last))
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for the infected population after day 10
    term9 = torch.where(torch.arange(len(I_hat)) >= 10, 
                        aggregation_func(norm_func(I_hat[torch.arange(len(I_hat)) >= 10] - I_pred[torch.arange(len(I_hat)) >= 10])), 
                        torch.tensor(0))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0) + regul * term9
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for the infected population after day 10
    term9 = torch.where(torch.arange(len(I_hat)) >= 10, 
                        aggregation_func(norm_func(I_hat[torch.arange(len(I_hat)) >= 10] - I_pred[torch.arange(len(I_hat)) >= 10])), 
                        torch.tensor(0))
    
    # Define the remaining terms
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(D_hat - D_pred))
    term8 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0) + regul * term9
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    
    # Introduce a new penalty for days after day 10
    quarantine_penalty = 0.01
    last_infected_penalty_after_day_10 = 0.02
    
    loss_before_day_10 = regul * (term1 + term2) + (1 - regul) * (aggregation_func(norm_func(f2)))
    
    # Calculate the difference between predicted and actual infected cases after day 10
    I_pred_diff_after_day_10 = aggregation_func(norm_func(I_pred[10:] - I_hat[10:]))
    
    loss_after_day_10 = regul * term1 + quarantine_penalty * I_pred_diff_after_day_10
    
    # Combine the losses before and after day 10, with a higher weight on the last infected penalty after day 10
    loss = (loss_before_day_10 * 0.7) + (loss_after_day_10 * 0.3) + (last_infected_penalty * norm_func(I_pred_last-0)) + (last_infected_penalty_after_day_10 * I_pred_diff_after_day_10)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 # Increased the penalty for the last infected population to reflect the effect of mask regime after day 15.
    aggregation_func = torch.max # Replaced mean with max to ensure that the model reflects the effect of mask regime after day 15.
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 
    aggregation_func = torch.max 
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5, 6, 7 and 8
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last)) # Added this line to define term7
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 10.0  # Increased penalty to reflect the effect of mask regime after day 15
    aggregation_func = torch.max  # Replaced mean with max to consider the worst case scenario
    norm_func = lambda x: (x - 0.5) ** 2  # Shifted the mean to consider the effect of mask regime
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term2 + term4) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.6
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(torch.clamp_min(norm_func(I_hat - I_pred), 0)) # Introduce a minimum value of 0 to avoid negative terms due to mask regime effect.
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (aggregation_func(norm_func(f1)) + aggregation_func(norm_func(f2)) + aggregation_func(norm_func(f3)) + aggregation_func(norm_func(f4))) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 
    aggregation_func = torch.max # or other functions to prioritize the last observation
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) # term5, 6, 7, 8 have been removed
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15 
    aggregation_func = torch.max # or other functions to prioritize the last observation
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define the missing terms
    term5 = aggregation_func(norm_func(f1 - f2))  # assuming f1 and f2 are some variables
    term6 = aggregation_func(norm_func(f3 - f4))  # assuming f3 and f4 are some variables
    term7 = aggregation_func(norm_func(I_hat - I_pred_last))
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.9  # Increased regularization strength after day 15
    last_infected_penalty = 0.07  # Adjusted penalty weight for the last infected population
    aggregation_func = torch.sum  # Changed to sum to give equal importance to each time step
    norm_func = torch.norm  # Changed to L2 norm

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) * (I_pred != 0)  # Mask out the last infected population after day 15
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    if I_pred_last.item() != 0:
        mask_regime_weight = 1  # Apply a weight to reflect the effect of the mask regime after day 15
    else:
        mask_regime_weight = 0

    loss = regul * (term1 + term3 + term4) + (1 - regul) * (term2 + f1.l1_loss() + f2.l1_loss() + f3.l1_loss() + f4.l1_loss()) 
    # Removed L1 penalty on last_infected_penalty as it's now being masked out
    loss += mask_regime_weight * torch.norm(I_pred_last, p=0)  # Add a term to reflect the effect of the mask regime after day 15
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.9  # Increased regularization strength after day 15
    last_infected_penalty = 0.07  # Adjusted penalty weight for the last infected population
    aggregation_func = torch.sum  # Changed to sum to give equal importance to each time step
    norm_func = torch.norm  # Changed to L2 norm

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) * (I_pred != 0)  # Mask out the last infected population after day 15
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    if I_pred_last.item() != 0:
        mask_regime_weight = 1  # Apply a weight to reflect the effect of the mask regime after day 15
    else:
        mask_regime_weight = 0

    loss = regul * (term1 + term3 + term4) + (1 - regul) * (term2 
        + torch.nn.L1Loss()(f1, torch.zeros_like(f1)) 
        + torch.nn.L1Loss()(f2, torch.zeros_like(f2)) 
        + torch.nn.L1Loss()(f3, torch.zeros_like(f3)) 
        + torch.nn.L1Loss()(f4, torch.zeros_like(f4)))
    # Removed L1 penalty on last_infected_penalty as it's now being masked out
    loss += mask_regime_weight * torch.norm(I_pred_last, p=0)  # Add a term to reflect the effect of the mask regime after day 15
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for the infected population after day 15
    last_infected_penalty_day_15 = 0.1
    term5 = torch.where(I_pred_last > 15, last_infected_penalty_day_15 * norm_func(I_pred_last-15), 
                        last_infected_penalty * norm_func(I_pred_last-0))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (aggregation_func(norm_func(f1)) +
                                                                  aggregation_func(norm_func(f2)) +
                                                                  aggregation_func(norm_func(f3)) +
                                                                  aggregation_func(norm_func(f4))) + term5
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for the difference between predicted and actual infected population after day 15
    last_infected_penalty_after_15 = 0.1
    I_pred_last_term = aggregation_func(norm_func(I_pred_last - I_hat[-16:]))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (aggregation_func(norm_func(f1)) + 
                                                                    aggregation_func(norm_func(f2)) +
                                                                    aggregation_func(norm_func(f3)) +
                                                                    aggregation_func(norm_func(f4))) + \
           last_infected_penalty * norm_func(I_pred_last-0) + last_infected_penalty_after_15 * I_pred_last_term
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for the infected population after day 10
    term9 = torch.where(torch.arange(len(I_hat)) >= 10, 
                        aggregation_func(norm_func(I_hat[torch.arange(len(I_hat)) >= 10] - I_pred[torch.arange(len(I_hat)) >= 10])), 
                        torch.zeros_like(aggregation_func(norm_func(I_hat - I_pred))))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * (norm_func(I_pred_last-0) + norm_func(term9))
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for the infected population after day 10
    term9 = torch.where(torch.arange(len(I_hat)) >= 10, 
                        aggregation_func(norm_func(I_hat[torch.arange(len(I_hat)) >= 10] - I_pred[torch.arange(len(I_hat)) >= 10])), 
                        torch.zeros_like(aggregation_func(norm_func(I_hat - I_pred))))
    
    # Define term5 to term8
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(D_hat - D_pred))
    term8 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * (norm_func(I_pred_last-0) + norm_func(term9))
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_day = torch.max(torch.tensor([10]))
    I_last_days = I_pred_last[:, last_day:]
    I_quarantine_penalty = aggregation_func(norm_func(I_last_days)) * quarantine_penalty
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0) + I_quarantine_penalty
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_day = torch.max(torch.tensor([10]))
    I_last_days = I_pred_last[:, last_day:]
    
    if len(I_last_days.shape) == 2:  # Check if it's a tensor with more than one dimension
        I_quarantine_penalty = aggregation_func(norm_func(I_last_days)) * quarantine_penalty
    else:
        I_quarantine_penalty = norm_func(I_last_days) * quarantine_penalty
    
    loss = regul * (term1 + term2 + term3 + term4) + last_infected_penalty * norm_func(I_pred_last-0) + I_quarantine_penalty
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_day = torch.max(torch.tensor([10]))
    I_last_days = I_pred_last[:, last_day:]
    
    if len(I_last_days.shape) == 2:  # Check if it's a tensor with more than one dimension
        I_quarantine_penalty = aggregation_func(norm_func(I_last_days)) * quarantine_penalty
    else:
        I_quarantine_penalty = norm_func(I_last_days) * quarantine_penalty
    
    loss = regul * (term1 + term2 + term3 + term4) + last_infected_penalty * norm_func(I_pred_last-0) + I_quarantine_penalty
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_day = torch.max(torch.tensor([10]))
    I_last_days = I_pred_last[:, last_day:]
    
    if len(I_last_days.shape) == 2:  # Check if it's a tensor with more than one dimension
        I_quarantine_penalty = aggregation_func(norm_func(I_last_days)) * quarantine_penalty
    else:
        I_quarantine_penalty = norm_func(I_last_days) * quarantine_penalty
    
    loss = regul * (term1 + term2 + term3 + term4) + last_infected_penalty * norm_func(I_pred_last-0) + I_quarantine_penalty
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.02
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Calculate the penalty for days after 10
    I_penalty_after_10 = aggregation_func(norm_func(I_pred_last))
    loss_after_10 = last_infected_penalty_after_10 * norm_func(I_penalty_after_10)
    
    # Combine the losses before and after day 10
    loss = regul * (loss_before_10) + (1 - regul) * (loss_after_10) + quarantine_penalty * norm_func(I_pred_last-0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.02

    loss_before_10 = regul * (term1 + term2 + term3 + term4)

    # Calculate the penalty for days after 10
    I_penalty_after_10 = aggregation_func(norm_func(I_pred_last))
    loss_after_10 = last_infected_penalty_after_10 * norm_func(I_penalty_after_10)

    # Combine the losses before and after day 10
    loss = regul * (loss_before_10) + (1 - regul) * (last_infected_penalty * norm_func(I_pred_last)) + quarantine_penalty * norm_func(I_pred_last-0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_day = torch.max(torch.tensor([10]))
    I_penalty = (I_pred_last - 0) * (last_day > 10).float()
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_penalty) + quarantine_penalty * norm_func((I_pred_last - 0) * (last_day > 10).float())
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_day = torch.max(torch.tensor([10]))
    I_penalty = (I_pred_last - 0) * (last_day > 10).float()

    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6) + last_infected_penalty * norm_func(I_penalty) + quarantine_penalty * norm_func((I_pred_last - 0) * (last_day > 10).float())
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # reduced penalty to allow for longer tail
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    loss +=  last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # reduced penalty to allow for longer tail
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5, 6, 7 and 8
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last)) 
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    loss +=  last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # reduced penalty to allow for longer tail
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5, 6, 7 and 8
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last)) 
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Add last_infected_penalty only if I_pred_last is not empty
    if len(I_pred_last) > 0:
        loss +=  last_infected_penalty * norm_func(I_pred_last[0]-0)

    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # decreased the penalty to make the tail longer
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    loss +=  last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # decreased the penalty to make the tail longer
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5, 6, 7 and 8
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last)) 
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    loss +=  last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # decreased the penalty to make the tail longer
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5, 6, 7 and 8
    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f3 - f4))
    term7 = aggregation_func(norm_func(I_hat - I_pred_last)) 
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Add last_infected_penalty only if I_pred_last is not empty
    if len(I_pred_last) > 0:
        loss +=  last_infected_penalty * norm_func(I_pred_last[0]-0)

    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # decreased the penalty for a sharp decline after the peak 
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    loss +=  last_infected_penalty * norm_func(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.005 # reduced penalty to make the decline less sharp
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.005 
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5, 6, 7 and 8
    term5 = aggregation_func(norm_func(f1 - f2)) 
    term6 = aggregation_func(norm_func(f3 - f4)) 
    term7 = aggregation_func(norm_func(I_hat - I_pred))
    term8 = aggregation_func(norm_func(D_hat - D_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.01 # decreased the penalty to make the decline less sharp
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) 
    loss += last_infected_penalty * norm_func(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    # Introduce a penalty for the difference between predicted and actual infected cases after 5 days
    if I_pred.shape[0] > 5:
        last_infected_penalty_term = aggregation_func(norm_func(I_hat[-5:] - I_pred[-5:]))
        loss += last_infected_penalty * last_infected_penalty_term
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5 to 8
    term5 = aggregation_func(norm_func(f1 - f2))  # Assuming f1 and f2 are tensors of the same shape as I_hat and I_pred
    term6 = aggregation_func(norm_func(f3 - f4))  # Assuming f3 and f4 are tensors of the same shape as D_hat and D_pred
    term7 = aggregation_func(norm_func(f5 - f6))  # Assuming f5 and f6 are tensors of the same shape as R_hat and R_pred
    term8 = aggregation_func(norm_func(f7 - f8))  # Assuming f7 and f8 are tensors of the same shape as S_hat and S_pred

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)

    # Introduce a penalty for the difference between predicted and actual infected cases after 5 days
    if I_pred.shape[0] > 5:
        last_infected_penalty_term = aggregation_func(norm_func(I_hat[-5:] - I_pred[-5:]))
        loss += last_infected_penalty * last_infected_penalty_term

    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    # Define terms 5 to 8
    term5 = aggregation_func(norm_func(f1 - f2))  
    term6 = aggregation_func(norm_func(f3 - f4))
    
    # Assuming R_hat and R_pred are tensors of the same shape as S_hat and S_pred
    term7 = aggregation_func(norm_func(R_hat - R_pred))  # 
    term8 = aggregation_func(norm_func(S_hat - S_pred))

    loss = regul * (aggregation_func(norm_func(S_hat - S_pred)) +
                    aggregation_func(norm_func(I_hat - I_pred)) +
                    aggregation_func(norm_func(D_hat - D_pred)) +
                    aggregation_func(norm_func(R_hat - R_pred))) + \
           (1 - regul) * (term5 + term6 + term7 + term8)

    # Introduce a penalty for the difference between predicted and actual infected cases after 5 days
    if I_pred.shape[0] > 5:
        last_infected_penalty_term = aggregation_func(norm_func(I_hat[-5:] - I_pred[-5:]))
        loss += last_infected_penalty * last_infected_penalty_term

    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 := aggregation_func(norm_func(f1))) + \
           (1 - regul) * (term6 := aggregation_func(norm_func(f2))) + \
           (1 - regul) * (term7 := aggregation_func(norm_func(f3))) + \
           (1 - regul) * (term8 := aggregation_func(norm_func(f4))) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05 / (7 * 10)
    aggregation_func = torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05 / (7 * 10)
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5 to 8
    f1_last = torch.mean(torch.square(f1[-1] - f2[-1]))
    f2_last = torch.mean(torch.square(f2[-1] - f3[-1]))
    f3_last = torch.mean(torch.square(f3[-1] - f4[-1]))
    f4_last = torch.mean(torch.square(f4[-1] - 0)) # Assuming the last term is zero

    term5 = aggregation_func(norm_func(f1 - f2))
    term6 = aggregation_func(norm_func(f2 - f3))
    term7 = aggregation_func(norm_func(f3 - f4))
    term8 = aggregation_func(norm_func(f4 - 0))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define term5, term6, term7 and term8
    term5 = aggregation_func(norm_func(f1))  # f1 is the equation for the change in the susceptible population (dS/dt)
    term6 = aggregation_func(norm_func(f2))  # f2 is the equation for the change in the infected population (dI/dt)
    term7 = aggregation_func(norm_func(f3))  # f3 is the equation for the change in the recovered population (dR/dt)
    term8 = aggregation_func(norm_func(f4))  # f4 is the equation for the change in the deceased population (dD/dt)

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05 / (I_pred_last.item() + 1e-9)
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
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
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
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 10
    aggregation_func = torch.max
    norm_func = torch.abs
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 10
    aggregation_func = torch.max
    norm_func = torch.abs

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define and calculate the remaining terms
    term5 = aggregation_func(norm_func(f1 - f1.detach()))
    term6 = aggregation_func(norm_func(f2 - f2.detach()))
    term7 = aggregation_func(norm_func(f3 - f3.detach()))
    term8 = aggregation_func(norm_func(f4 - f4.detach()))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.15
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Calculate the difference between predicted and actual infected cases for days after 10
    I_pred_after_10 = I_pred[10:]
    I_hat_after_10 = I_hat[10:]
    last_infected_penalty_after_10_term = aggregation_func(norm_func(I_pred_after_10 - I_hat_after_10))
    
    loss_after_10 = regul * (term1[:10] + term2[:10] + term3[:10] + term4[:10]) + (1 - regul) * (term5[:10] + term6[:10] + term7[:10] + term8[:10])
    
    # Combine the two losses
    loss = loss_before_10 + loss_after_10 + quarantine_penalty * last_infected_penalty_after_10_term + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.15
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(I_pred_last-0)) # Define term5 to term8 here

    # Calculate the difference between predicted and actual infected cases for days after 10
    I_pred_after_10 = I_pred[10:]
    I_hat_after_10 = I_hat[10:]
    last_infected_penalty_after_10_term = aggregation_func(norm_func(I_pred_after_10 - I_hat_after_10))
    
    loss_after_10 = regul * (term1[:10] + term2[:10] + term3[:10] + term4[:10]) + (1 - regul) * last_infected_penalty_after_10_term # Use last_infected_penalty_after_10_term instead of term5 to term8

    # Combine the two losses
    loss = loss_before_10 + loss_after_10 + quarantine_penalty * last_infected_penalty_after_10_term + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.15
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(I_pred_last-0)) # Define term5 to term8 here

    # Calculate the difference between predicted and actual infected cases for days after 10
    I_pred_after_10 = I_pred[10:]
    I_hat_after_10 = I_hat[10:]
    
    if len(I_hat_after_10) > 0:
        last_infected_penalty_after_10_term = aggregation_func(norm_func(I_pred_after_10 - I_hat_after_10))
    else:
        last_infected_penalty_after_10_term = torch.tensor(0, dtype=torch.float64)
    
    loss_after_10 = regul * (term1[:10] + term2[:10] + term3[:10] + term4[:10]) + (1 - regul) * last_infected_penalty_after_10_term # Use last_infected_penalty_after_10_term instead of term5 to term8

    # Combine the two losses
    loss = loss_before_10 + loss_after_10 + quarantine_penalty * last_infected_penalty_after_10_term + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.15
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(I_pred_last-0)) # Define term5 to term8 here

    # Calculate the difference between predicted and actual infected cases for days after 10
    I_pred_after_10 = I_pred[10:]
    I_hat_after_10 = I_hat[10:]
    
    if len(I_hat_after_10) > 0:
        last_infected_penalty_after_10_term = aggregation_func(norm_func(I_pred_after_10 - I_hat_after_10))
    else:
        last_infected_penalty_after_10_term = torch.tensor(0, dtype=torch.float64)
    
    loss_after_10 = regul * (term1[:10] + term2[:10] + term3[:10] + term4[:10]) + (1 - regul) * (last_infected_penalty * norm_func(I_pred_last-0)) # Use last_infected_penalty_after_10_term instead of term5 to term8

    # Combine the two losses
    loss = loss_before_10 + loss_after_10 + quarantine_penalty * last_infected_penalty_after_10_term 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Introduce a new penalty for days after 10
    quarantine_penalty = 0.1
    last_infected_penalty_after_10 = 0.15
    
    loss_before_10 = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (norm_func(I_pred_last-0)) 

    # Calculate the difference between predicted and actual infected cases for days after 10
    I_pred_after_10 = I_pred[10:]
    I_hat_after_10 = I_hat[10:]
    
    if len(I_hat_after_10) > 0:
        last_infected_penalty_after_10_term = aggregation_func(norm_func(I_pred_after_10 - I_hat_after_10))
    else:
        last_infected_penalty_after_10_term = torch.tensor(0, dtype=torch.float64)
    
    loss_after_10 = regul * (term1[:10] + term2[:10] + term3[:10] + term4[:10]) + (1 - regul) * (last_infected_penalty * norm_func(I_pred_last-0)) 

    # Combine the two losses
    if len(I_hat_after_10) > 0:
        loss = loss_before_10 + loss_after_10 + quarantine_penalty * last_infected_penalty_after_10_term 
    else:
        loss = loss_before_10 + loss_after_10 
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    
    # Introduce a penalty for the difference between predicted and actual infected cases after day 10
    term3 = aggregation_func(norm_func(D_hat - D_pred))  # Removed this line
    term4 = aggregation_func(norm_func(R_hat - R_pred))  # Removed this line
    
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    
    # Introduce a penalty for the difference between predicted and actual recovered cases after day 10
    term7 = aggregation_func(norm_func(f3))
    
    # Introduce a penalty for the difference between predicted and actual deceased cases after day 10
    term8 = aggregation_func(norm_func(f4))
    
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
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
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Introduce a new penalty for the infected population after day 10
    if I_pred_last >= 0:
        last_infected_penalty = 0.05
    else:
        last_infected_penalty = 1.0
    
    loss += last_infected_penalty * norm_func(I_pred_last)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
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
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Introduce a new penalty for the infected population after day 10
    if I_pred_last.item() >= 0:
        last_infected_penalty = 0.05
    else:
        last_infected_penalty = 1.0
    
    loss += last_infected_penalty * norm_func(I_pred_last)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
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
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Introduce a new penalty for the infected population after day 10
    if I_pred_last.item() >= 0:
        last_infected_penalty = 0.05
    else:
        last_infected_penalty = 1.0
    
    # Add a dummy dimension to I_pred_last to match the broadcast shape [1]
    loss += last_infected_penalty * norm_func(I_pred_last.unsqueeze(0))
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
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
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Introduce a new penalty for the infected population after day 10
    if I_pred_last.item() >= 0:
        last_infected_penalty = 0.05
    else:
        last_infected_penalty = 1.0
    
    # Add a dummy dimension to I_pred_last to match the broadcast shape [1]
    loss += last_infected_penalty * norm_func(I_pred_last.unsqueeze(0).unsqueeze(-1))
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    loss = regul * (term1 + term2 + term3 + term4) 
    # Removed terms 5,6,7,8 as per your request
    loss += last_infected_penalty * norm_func(I_pred_last)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    loss = regul * (term1 + term2 + term3 + term4) 
    # Removed terms 5,6,7,8 as per your request

    if isinstance(I_pred_last, torch.Tensor):
        I_pred_last = I_pred_last.unsqueeze(0)
    
    loss += last_infected_penalty * norm_func(I_pred_last)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    loss = regul * (term1 + term2 + term3 + term4) 

    if isinstance(I_pred_last, torch.Tensor):
        I_pred_last = I_pred_last.unsqueeze(0)
    
    # Add a new dimension to I_pred_last for broadcasting
    I_pred_last = I_pred_last.unsqueeze(-1)

    loss += last_infected_penalty * norm_func(I_pred_last)
    
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a penalty for the difference in infection rates after day 10
    term5 = aggregation_func(torch.where(f2 > 0, norm_func((f1[:, 10:] - f1[:10]) / (f2[:, 10:] + 1e-8)), torch.tensor(0.)))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a penalty for the difference in infection rates after day 10
    f1_slice_1 = f1[:, :10]  # get first 10 elements of each row
    f2_slice_1 = f2[:, :10]  # get first 10 elements of each row
    f1_slice_2 = f1[:, 10:]  # get last elements of each row
    f2_slice_2 = f2[:, 10:]  # get last elements of each row
    
    term5 = aggregation_func(norm_func((f1_slice_2 - f1_slice_1) / (f2_slice_2 + 1e-8)))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5) + last_infected_penalty * norm_func(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a penalty for the difference in infection rates after day 10
    f1_slice_1 = f1[:, :10]  # get first 10 elements of each row
    f2_slice_1 = f2[:, :10]  # get first 10 elements of each row
    f1_slice_2 = f1[:, 10:]  # get last elements of each row
    f2_slice_2 = f2[:, 10:]  # get last elements of each row
    
    term5 = aggregation_func(norm_func((f1_slice_2 - f1_slice_1) / (torch.clamp(f2_slice_2, min=1e-8))))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5) + last_infected_penalty * norm_func(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if len(I_hat) > 10 else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if len(S_hat) > 10 else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term1+term2+term3+term4) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term1+term2+term3+term4) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a 14-day lag for the formation of immunity
    term5 = aggregation_func(norm_func(f1))  # No change in this term
    term6 = aggregation_func(norm_func(f2))  # No change in this term
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    # The 14-day lag is not directly applicable to the last_infected_penalty term, 
    # so we will remove it for now
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    
    # Introduce a new term to account for the 14-day lag in immunity formation
    R_hat_lag = R_hat[-14:]  # Assuming R_hat has at least 14 elements
    R_pred_lag = R_pred[-14:]
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    
    term4 = aggregation_func(norm_func(R_hat_lag - R_pred_lag))  # Use the lagged values
    
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Introduce a new penalty for the infected population after day 10
    if I_pred.shape[0] > 10:
        last_infected_penalty_after_10 = 0.2
        loss += last_infected_penalty_after_10 * norm_func(I_hat - I_pred)
        
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define the remaining terms
    term5 = aggregation_func(norm_func(f1 - f1.detach()))
    term6 = aggregation_func(norm_func(f2 - f2.detach()))
    term7 = aggregation_func(norm_func(f3 - f3.detach()))
    term8 = aggregation_func(norm_func(f4 - f4.detach()))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)

    # Introduce a new penalty for the infected population after day 10
    if I_pred.shape[0] > 10:
        last_infected_penalty_after_10 = 0.2
        loss += last_infected_penalty_after_10 * norm_func(I_hat - I_pred)
        
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define the remaining terms
    term5 = aggregation_func(norm_func(f1 - f1.detach()))
    term6 = aggregation_func(norm_func(f2 - f2.detach()))
    term7 = aggregation_func(norm_func(f3 - f3.detach()))
    term8 = aggregation_func(norm_func(f4 - f4.detach()))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)

    # Introduce a new penalty for the infected population after day 10
    if I_pred.shape[0] > 10:
        last_infected_penalty_after_10 = 0.2
        # Check if I_hat and I_pred have the same shape before subtracting them
        if I_hat.shape == I_pred.shape:
            loss += last_infected_penalty_after_10 * norm_func(I_hat - I_pred)
        else:
            # If they don't have the same shape, broadcast I_hat to match the shape of I_pred
            loss += last_infected_penalty_after_10 * norm_func(aggregation_func(I_hat.unsqueeze(0)) - I_pred)

    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    
    # Introduce a new penalty for the infected population after day 10
    last_10_infected_penalty = 0.5 if I_pred_last > 10 else 0
    
    loss = regul * (term1 + term2) + (1 - regul) * (aggregation_func(norm_func(f1)) + aggregation_func(norm_func(f2))) + \
           last_infected_penalty * norm_func(I_pred_last-0) + last_10_infected_penalty * norm_func((I_hat - I_pred).abs())
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
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
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    if I_pred_last != 0:
        loss += last_infected_penalty * norm_func(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
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
    
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8)
    
    # Check if I_pred_last is not zero and has a shape of []
    if I_pred_last.shape == [] and I_pred_last != 0:
        loss += last_infected_penalty * norm_func(I_pred_last)
        
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred_last > 10) else torch.mean
    norm_func = torch.square
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred_last > 10) else torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5 to 8
    term5 = aggregation_func(norm_func(f1 - f1.detach()))
    term6 = aggregation_func(norm_func(f2 - f2.detach()))
    term7 = aggregation_func(norm_func(f3 - f3.detach()))
    term8 = aggregation_func(norm_func(f4 - f4.detach()))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.max if (I_pred_last > 10) else torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    # Define terms 5 to 8
    term5 = aggregation_func(norm_func(f1 - f1.detach()))
    term6 = aggregation_func(norm_func(f2 - f2.detach()))
    term7 = aggregation_func(norm_func(f3 - f3.detach()))
    term8 = aggregation_func(norm_func(f4 - f4.detach()))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 + term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last)
    
    # Make the loss a scalar
    return loss.mean()

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    #term3 = aggregation_func(norm_func(D_hat - D_pred)) # Removed this line
    term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    #term7 = aggregation_func(norm_func(f3)) # Removed this line
    #term8 = aggregation_func(norm_func(f4)) # Removed this line
    
    loss = regul * (term1 + term2 + term4) + (1 - regul) * (term5 + term6) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    #term3 = aggregation_func(norm_func(D_hat - D_pred)) 
    #term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    #term7 = aggregation_func(norm_func(f3))
    #term8 = aggregation_func(norm_func(f4))
    
    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square
    
    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    
    # Reduce the weight of the infection rate in the forecast
    loss = regul * (term1 + term2) + (0.4 - regul) * (aggregation_func(norm_func(f2))) + last_infected_penalty * torch.square(I_pred_last)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))
    # term3 = aggregation_func(norm_func(D_hat - D_pred))
    # term4 = aggregation_func(norm_func(R_hat - R_pred))
    
    # Introduce a penalty for the infected population not decreasing after 20 days
    if I_pred_last.item() > I_hat.item():
        last_infected_penalty = 0.1
    
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))

    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6) + last_infected_penalty * (I_hat - I_pred).abs()
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))

    # Introduce a penalty for the infected population not decreasing after 20 days
    if I_pred_last.item() > I_hat.item():
        last_infected_penalty = 0.1
    
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))

    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6)
    
    # Add a penalty for the infected population not decreasing after 20 days
    if I_pred_last.item() > I_hat.item():
        loss += last_infected_penalty * (I_hat - I_pred).abs()
        
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred))

    # Introduce a penalty for the infected population not decreasing after 20 days
    if I_pred_last.item() == I_hat.item():
        last_infected_penalty = 0.05
    
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))

    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6)
    
    # Add a penalty for the infected population not decreasing after 20 days
    if I_pred_last.item() == I_hat.item():
        loss += last_infected_penalty * (I_hat - I_pred).abs()
        
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + (I_pred_last - I_hat).abs().mean() * last_infected_penalty  # added penalty for not decreasing infected population after 20 days
    # term3 = aggregation_func(norm_func(D_hat - D_pred))
    # term4 = aggregation_func(norm_func(R_hat - R_pred))
    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))

    loss = regul * (term1 + term2) + (1 - regul) * (term5 + term6)
    return loss

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

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    time_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + time_penalty * (I_pred_last - I_pred)
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8)
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    time_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + time_penalty * (t - t_last)**2 * norm_func(I_hat - I_pred)
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

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    time_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + time_penalty * (t_last - t_last)**2 * norm_func(I_hat - I_pred)
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

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    time_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + time_penalty * (t_last - t)**2 * norm_func(I_hat - I_pred)
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

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    time_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + time_penalty * torch.sum(torch.abs((torch.arange(len(I_hat))[:, None] - len(I_hat) // 2) * (I_hat - I_pred)))
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

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    time_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + time_penalty * torch.sum(torch.abs((torch.arange(len(I_hat))[:, None] - len(I_hat) // 2) * (I_hat - I_pred)))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + \
        last_infected_penalty * norm_func(I_pred_last-0)
    
    # Add a penalty for the time dynamics of the infected population
    if len(I_hat) > 20:
        term9 = aggregation_func(norm_func((I_hat[1:] - I_hat[:-1])))
        loss += 0.2 * term9
    
    return loss

# deepseek (comment 1 class 1.1)
def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    tail_length_penalty = 0.1  # Added penalty for tail length

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + tail_length_penalty * \
        torch.mean(torch.relu(-f2[20:]))  # Modified term2
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

# perplexity (comment 1 class 1.1)
def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    expert_penalty = 0.1  # added penalty coefficient for term2 based on expert comment

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

    loss = regul * (term1 + (1 + expert_penalty) * term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + \
        last_infected_penalty * norm_func(I_pred_last - 0)
    return loss

# deepseek (comment align class 4.1)
def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05
    peak_offset = 0.1  # Added offset for peak alignment

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + peak_offset
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

import torch

# perplexity (comment align class 1.1)
def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):

    regul = 0.8
    last_infected_penalty = 0.05

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

    # Expert comment: after the peak there should be a sharper decline in the number of infected
    peak_index = torch.argmax(I_hat)
    decline_penalty = 0.1 * \
        torch.mean(norm_func(I_pred[peak_index:] - I_hat[peak_index:]))

    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + \
        last_infected_penalty * norm_func(I_pred_last-0) + decline_penalty
    return loss

# after perplexity


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):

    regul = 0.8
    last_infected_penalty = 0.05

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

    # Expert comment: after the peak there should be a sharper decline in the number of infected
    peak_index = torch.argmax(I_hat)
    desired_peak_offset = 5  # Offset for the desired peak day
    decline_penalty = 0.1 * \
        torch.mean(norm_func(
            I_pred[peak_index + desired_peak_offset:] - I_hat[peak_index + desired_peak_offset:]))

    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + \
        last_infected_penalty * norm_func(I_pred_last-0) + decline_penalty
    return loss



# after2 perplexity
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
    peak_ratio_penalty = 10.0 * norm_func(peak_infected_pred / (peak_infected_true + 1e-8) - 1.05)
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
            I_pred[peak_index + desired_peak_offset:] - (I_hat[peak_index + desired_peak_offset:] * 1.2))) # sharper decline

    loss = regul * (term1 + term2 + term3 + term4) + \
        (1 - regul) * (term5 + term6 + term7 + term8) + \
        last_infected_penalty * norm_func(I_pred_last - 0) + decline_penalty
    return loss

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.05

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(1.5 * norm_func(I_hat - I_pred)) # Increased penalty for infected population
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

import torch


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):
    regul = 0.8
    last_infected_penalty = 0.15

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(1.5 * norm_func(I_hat - I_pred)) # Increased penalty for infected population
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

# claude_2


def loss_dinn(S_hat, S_pred, I_hat, I_pred, D_hat, D_pred, R_hat, R_pred, f1, f2, f3, f4, I_pred_last):

    regul = 0.8
    last_infected_penalty = 0.05
    decline_penalty = 0.1

    aggregation_func = torch.mean
    norm_func = torch.square

    term1 = aggregation_func(norm_func(S_hat - S_pred))
    term2 = aggregation_func(norm_func(I_hat - I_pred)) + \
        decline_penalty * aggregation_func(torch.relu(f2))
    term3 = aggregation_func(norm_func(D_hat - D_pred))
    term4 = aggregation_func(norm_func(R_hat - R_pred))

    term5 = aggregation_func(norm_func(f1))
    term6 = aggregation_func(norm_func(f2))
    term7 = aggregation_func(norm_func(f3))
    term8 = aggregation_func(norm_func(f4))

    loss = regul * (term1 + term2 + term3 + term4) + (1 - regul) * (term5 +
                                                                    term6 + term7 + term8) + last_infected_penalty * norm_func(I_pred_last-0)
    return loss


# claude_3


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
