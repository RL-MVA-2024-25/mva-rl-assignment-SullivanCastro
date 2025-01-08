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
    from fast_env import FastHIVPatient as HIVPatient

    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=100
    ) 

    optim_hyperparameters = {
        "nb_neurons": 2**trial.suggest_int("nb_neurons", 2, 10),
        "gamma": trial.suggest_float("gamma", 0.95, 0.99),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2),
    }

    model = ProjectAgent()
    for optim in optim_hyperparameters:
        setattr(model, optim, optim_hyperparameters[optim])
    setattr(model, 'verbose', False)
    setattr(model, 'is_updating', False)
    setattr(model, 'path', 'model/grid_search.pth')
    _, _, _, _, validation_score = model.train(env, 100, False)

    return validation_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
