import optuna
import torch 
import torch.nn as nn
import numpy as np
from train import ProjectAgent
# tune the dqn model with optuna
"""
config = {'nb_actions':env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 20,
          'path': 'model/dqn.pth',}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    n_action = env.action_space.n 
    nb_neurons=24
    DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, nb_neurons),
                            nn.ReLU(), 
                            nn.Linear(nb_neurons, n_action)).to(device)
    
    model = dqn_agent(config, DQN)
"""

def objective(trial):
    from gymnasium.wrappers import TimeLimit
    from env_hiv import HIVPatient

    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
    ) 

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    buffer_size = trial.suggest_int("buffer_size", 1000, 1000000)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.1)
    epsilon_max = trial.suggest_float("epsilon_max", 0.1, 1.0)
    epsilon_decay_period = trial.suggest_int("epsilon_decay_period", 10, 200)
    epsilon_delay_decay = trial.suggest_int("epsilon_delay_decay", 10, 200)
    batch_size = trial.suggest_int("batch_size", 10, 100)
    nb_gradient_steps = trial.suggest_int("nb_gradient_steps", 1, 10)
    path = "model/dqn.pth"

    config = {'nb_actions':env.action_space.n,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'buffer_size': buffer_size,
            'epsilon_min': epsilon_min,
            'epsilon_max': epsilon_max,
            'epsilon_decay_period': epsilon_decay_period,
            'epsilon_delay_decay': epsilon_delay_decay,
            "nb_gradient_steps": nb_gradient_steps,
            'batch_size': batch_size,
            'path': path,
            "verbose": False}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    n_action = env.action_space.n
    nb_neurons = trial.suggest_int("nb_neurons", 10, 100)
    DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                            nn.ReLU(),
                            nn.Linear(nb_neurons, nb_neurons),
                            nn.ReLU(), 
                            nn.Linear(nb_neurons, n_action)).to(device)
    model = ProjectAgent(config, DQN)
    episode_return, _, _, _ = model.train(env, 200, False)

    return episode_return[-1]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
