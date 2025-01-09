from gymnasium.wrappers import TimeLimit
# from env_hiv import HIVPatient
from fast_env import FastHIVPatient as HIVPatient
from evaluate import evaluate_HIV

import yaml
import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class ProjectAgent:
    def load(self):
        # save the model
        import os
        if not os.path.exists(self.path):
            self.path = os.path.join(os.getcwd(), self.path)
        print("Loading model from ", self.path)
        self.model.load_state_dict(torch.load(self.path))

    def save(self, final=False):
        if final:
            print("Saving final model to ", self.path.replace(".pth", "_final.pth"))
            torch.save(self.model.state_dict(), self.path.replace(".pth", "_final.pth"))
        else:
            print("Saving best model to ", self.path)
            torch.save(self.best_model.state_dict(), self.path)

    def act(self, state):
        return greedy_action(self.best_model, state)
    
    def __init__(self, config=None, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if config is None:
            # read the config.yaml file
            with open("src/config.yaml", "r") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.nb_actions = env.action_space.n
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop        
        self.criterion = torch.nn.SmoothL1Loss()
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 1
        self.path = config['path'] if 'path' in config.keys() else 'model/base_model.pth'
        self.verbose = config['verbose'] if 'verbose' in config.keys() else True
        self.is_updating = config['is_updating'] if 'is_updating' in config.keys() else True

        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 
        nb_neurons=config['nb_neurons'] if 'nb_neurons' in config.keys() else 64
        self.model = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action) # 5 layers
            ).to(self.device)
        if model_path is not None:
            self.path = model_path
            self.load()
        self.best_model = deepcopy(self.model).to(self.device)
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def MC_eval(self, env, nb_trials):   
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials, device): 
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.best_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   
        MC_avg_discounted_reward = []   
        V_init_state = []   
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        previous_best = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.best_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.best_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.best_model.load_state_dict(target_state_dict)
            # next transition
            step += 1

            if done or trunc:
                episode += 1
                
                val_score = evaluate_HIV(agent=self, nb_episode=5)
                                
                if self.verbose:
                    print(f"Episode {episode:3d} | "
                        f"Epsilon {epsilon:6.2f} | "
                        f"Batch Size {len(self.memory):5d} | "
                        f"Episode Return {episode_cum_reward:.2e} | "
                        f"Score on HIV {val_score:.2e}")
                state, _ = env.reset()

                if val_score > previous_best:
                    previous_best = val_score
                    self.best_model = deepcopy(self.model).to(self.device)
                    if self.is_updating:
                        self.save()
                episode_return.append(episode_cum_reward)
                
                episode_cum_reward = 0
            else:
                state = next_state

        self.save(final=True)
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state, previous_best
    

if __name__ == "__main__":
    from gymnasium.wrappers import TimeLimit
    from fast_env import FastHIVPatient as HIVPatient

    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
    )  # The time wrapper limits the number of steps in an episode at 200.
    
    model = ProjectAgent()
    model.train(env, 200)

