from Policy import Actor
import gymnasium as gym
import torch
import numpy as np
from policy_evaluation import evaluate_policy


    
def zeroth_order(policy, env, episodes, alpha, alpha_decay, evaluation_episodes, device, log_file_name, sigma, sigma_decay, discount_factor):

    best_perturbation = None
    best_reward = -float('inf')

    min_sigma = 0.01


    with open(log_file_name, 'w') as log_file:
        for episode in range(episodes):
            
            perturbation = [sigma * torch.randn_like(p).to(device) for p in policy.parameters()]

            # Evaluate negative and positive perturbation
            score_pos = evaluate_policy(policy, env, perturbation, True, evaluation_episodes, device, discount_factor)
            score_neg = evaluate_policy(policy, env, perturbation, False, evaluation_episodes, device, discount_factor)

            

            # Calculate gradient and update parameters
            gradient = [0.5 * (score_pos - score_neg) * pert for pert in perturbation]

            with torch.no_grad():
                for param, grad in zip(policy.parameters(), gradient): 
                    param += alpha * grad

            

            print(f"Episode {episode + 1}/{episodes}, Score: {max(score_pos, score_neg)}")
            print('RETURN', episode + 1, max(score_pos, score_neg) , file=log_file)  
            
            
            sigma = max(min_sigma, sigma * sigma_decay)

    env.close() 
