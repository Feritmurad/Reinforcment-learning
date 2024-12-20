import torch
import numpy as np


def evaluate_policy(policy, env, perturbation, positive, evaluation_episodes, device,  discount_factor=0.998, max_steps_per_episode=1000):

    # Save original parameters
    old_parameters = [param.clone() for param in policy.parameters()]

    # Apply perturbation
    with torch.no_grad():
        for param, pert in zip(policy.parameters(), perturbation):
            if positive:
                param += pert
            else:
                param -= pert

    total_reward = 0
    for _ in range(evaluation_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0 
        episode_reward = 0
        discount = 1.0
        while not done and step_count < max_steps_per_episode :
            state_tensor = torch.from_numpy(state).float().to(device)
            action = policy(state_tensor).cpu().detach().numpy()
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward * discount
            discount *= discount_factor
            step_count += 1
        total_reward += episode_reward

    # Restore parameters
    with torch.no_grad():
        for old_param, param in zip(old_parameters, policy.parameters()):
            param.copy_(old_param)

    return total_reward / evaluation_episodes