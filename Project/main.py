from Policy import Actor
import gymnasium as gym
import torch
import numpy as np
from zeroth_order import *
from population_methods import *

def zeroth_order_grid_search():
    # Initialize the environment and the policy network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = Actor(state_dim, action_dim).to(device)

    # Define the grid search parameters
    evaluation_episodes = 1
    episodes = [1000]
    alpha_values = [0.01]
    alpha_decay_values = [0.999]
    sigma_values = [0.1]
    sigma_decay_values = [0.999]
    discount_factor= 0.999

    for alpha in alpha_values:
        for alpha_decay in alpha_decay_values:
            for sigma in sigma_values:
                for sigma_decay in sigma_decay_values:
                    for episode in episodes:
                        # Create a new policy for each set of hyperparameters
                        policy = Actor(state_dim, action_dim).to(device)
                        log_file_name = f'zeroth_order_log_alpha_{alpha}_log_alpha_decay_{alpha_decay}_sigma{sigma}_sigma_decay{sigma_decay}_discount_factor{discount_factor}_epiosde{episode}.txt'
                        print(f"Zeroth Order Training with alpha={alpha},alpha_decay={alpha_decay},sigma={sigma},sigma_decay={sigma_decay},discount_factor{discount_factor}")
                        zeroth_order(policy, env, episode, alpha, alpha_decay, evaluation_episodes, device, log_file_name, sigma, sigma_decay, discount_factor)


def population_methods_grid_search():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = Actor(state_dim, action_dim).to(device)
    
    episodes = 300
    perturbations_amount = [30]
    evaluation_episodes = [1,10]
    sigma_values = [0.1]
    sigma_decay_values = [0.999]
    discount_factor= 0.999

    for pert_a in perturbations_amount:
        for sigma in sigma_values:
            for sigma_decay in sigma_decay_values:
                for ev_episodes in evaluation_episodes:
                    # Create a new policy for each set of hyperparameters
                    policy = Actor(state_dim, action_dim).to(device)
                    log_file_name = f'population_method_perturbations_amount_{pert_a}_discount_factor_{discount_factor}_sigma{sigma}_sigma_decay{sigma_decay}_evaluation_episodes{ev_episodes}.txt'
                    print(f"Population Method Training with perturbations_amount={pert_a},discount_factor={discount_factor},sigma={sigma},sigma_decay={sigma_decay},evaluation_episodes={ev_episodes}")
                    population_methods(policy, env, episodes, ev_episodes, device, log_file_name, pert_a, sigma, sigma_decay, discount_factor)

def main():
    zeroth_order_grid_search()
    population_methods_grid_search()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = Actor(state_dim, action_dim).to(device)
    
    zeroth_order(policy, env, episodes=1000, alpha=0.01, 
                 evaluation_episodes=10, device=device, log_file_name="zeroth order.txt", 
                 epsilon=1, epsilon_decay=0.997, sigma=0.5, sigma_decay=1, discount_factor= 0.998)
 
    population_methods(policy, env, episodes=300, evaluation_episodes=10, device=device, log_file_name="populations method.txt", pert_a=30, sigma=0.1, sigma_decay=0.999, discount_factor=0.999)
    

main()