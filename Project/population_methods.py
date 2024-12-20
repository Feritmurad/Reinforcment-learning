from Policy import Actor
import gymnasium as gym
import torch
import numpy as np
from policy_evaluation import evaluate_policy


def population_methods(policy, env, episodes, evaluation_episodes, device, log_file_name, perturbations_amount, sigma, sigma_decay, discount_factor):

    min_sigma = 0.01

    with open(log_file_name, 'w') as log_file:
        for episode in range(episodes):
            
            best_perturbation = None
            # Create and evaluate each perturbation
            best_score = -np.infty
            for i in range(perturbations_amount):
                
                # Create pertrubation
                perturbation = [torch.randn_like(p) * sigma for p in policy.parameters()]
                score = evaluate_policy(policy, env, perturbation, True, evaluation_episodes, device, discount_factor)

                # Choose best perturbation
                if score > best_score:
                    best_score = score
                    best_perturbation = perturbation
        
            with torch.no_grad():
                for param, pert in zip(policy.parameters(), best_perturbation):
                    param += pert


            print(f"Episode {episode + 1}/{episodes}, Score: {best_score}")
            print('RETURN', episode + 1, best_score, file=log_file)

            sigma = max(min_sigma, sigma * sigma_decay)
 

    env.close() 