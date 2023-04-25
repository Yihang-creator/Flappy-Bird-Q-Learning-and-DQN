import json
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


#QNetwork class and ReplayMemory class are implementation decisions.
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self,x):
        return self.model(x)


#Solution for question 8
class ReplayMemory(Dataset):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
        }

    def push(self, transition):
        if len(self.memory['state']) < self.capacity:
            for key, value in zip(self.memory.keys(), transition):
                self.memory[key].append(value)
        else:
            for key, value in zip(self.memory.keys(), transition):
                self.memory[key].pop(0)
                self.memory[key].append(value)

    def __len__(self):
        return len(self.memory['state'])

    def __getitem__(self, idx):
        return tuple(self.memory[key][idx] for key in self.memory.keys())

    def sample(self,batch_size):
        indices = random.sample(range(len(self.memory['state'])), batch_size)
        batch = [tuple(self.memory[key][idx] for key in self.memory.keys()) for idx in indices]
        return batch


# This class is for students to fill in. This class includes similar function to bot.py like
# act, update_scores, map_state. Load_qvalue and dump_qvalues are not necessary for grades.

# Solution for question 7
class DQNAgent(object):
    def __init__(self):
        self.gameCNT = 0  # Game count of current run, incremented after every death
        self.DUMPING_N = 25  # Number of iterations to dump Q values to JSON after
        self.TARGET_UPDATE = 10  # Number of episodes to update target network after
        self.discount = 1
        self.r = {0: 0.1, 1: -10}  # Reward function
        self.lr = 0.0001 # learning rate
        self.load_qvalues()
        self.last_state = None
        self.last_action = None
        self.batch_size = 128
        self.moves = []
        self.sigma = 0.5
        self.sigma_decay = 0.95
        self.sigma_min = 0.01
        self.memory = ReplayMemory(10000)



    def load_qvalues(self):
        self.policy_net = QNetwork(3, 2, 256)
        self.target_net = QNetwork(3, 2, 256)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        try:
            self.policy_net.load_state_dict(torch.load("data/qvalues.pt"))
            self.target_net.load_state_dict(torch.load("data/qvalues.pt"))
        except FileNotFoundError:
            print("Train from scratch")

    def remember(self,state,action,reward,next_state):
        self.memory.push([state,action,reward,next_state])

    def act(self, states_arr):
        """
        Chooses the best action with respect to the current state - Chooses 0 (don't flap) to tie-break
        """
        state = self.map_state(states_arr)

        if self.last_state is not None:
            self.moves.append(
                (self.last_state, self.last_action, state)
            )

        self.last_state = state  # Update the last_state with the current state

        with torch.no_grad():
            q_values = self.policy_net.forward(state)

        if torch.rand(1)[0] < self.sigma:
            self.last_action = torch.randint(low=0,high=2,size=(1,))[0]
        else:
            self.last_action = q_values.argmax().item()
        self.sigma = max(self.sigma_min,self.sigma*self.sigma_decay)
        return self.last_action

    def update_scores(self, dump_qvalues=True):

        history = list(reversed(self.moves))

        # Flag if the bird died in the top pipe
        high_death_flag = True if int(history[0][2][1]) > 120 else False

        # Q-learning score updates
        t = 1
        for exp in history:

            state = exp[0]
            act = exp[1]
            res_state = exp[2]

            # experience replay

            # Select reward
            if t == 1 or t == 2:
                self.remember(state,act,self.r[1],res_state)
            elif high_death_flag and act:
                self.remember(state,act,self.r[1],res_state)
                high_death_flag = False
            else:
                self.remember(state,act,self.r[0],res_state)

        if self.memory.__len__() < self.batch_size:
            return

        for i in range(100):
            minibatch = self.memory.sample(min(self.batch_size, len(self.memory)))
            state_batch, action_batch, reward_batch, next_state_batch= zip(*minibatch)

            # Convert the minibatch data to tensors
            state_batch = torch.stack(state_batch)
            action_batch = torch.tensor(action_batch)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
            next_state_batch = torch.stack(next_state_batch)

            #compute Q values for the current state-action pairs
            q_values = self.policy_net.forward(state_batch).gather(1,action_batch.unsqueeze(1)).squeeze(1)

            # compute the target Q values for the next states
            with torch.no_grad():
                target_q_values = self.target_net.forward(next_state_batch).max(1)[0]

            #Compute the expected Q values based on rewards and discounted future rewards
            expected_q_values = reward_batch + self.discount*target_q_values

            loss = nn.SmoothL1Loss()(q_values,expected_q_values)
            #Optimize the policy network
            self.optimizer.zero_grad()
            loss.backward()
            torch,nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
            self.optimizer.step()

        self.gameCNT += 1  # increase game count
        if dump_qvalues and self.gameCNT % self.DUMPING_N == 0:
            self.dump_qvalues()  # Dump weights for q-network (if game count % DUMPING_N == 0)

        if self.gameCNT % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.moves = []

    def map_state(self, states_arr):
        playerx = states_arr[0]
        playery = states_arr[1]
        vel = states_arr[2]
        lowerPipes = states_arr[3]
        if -playerx + lowerPipes[0]["x"] > -50:
            curPipe = lowerPipes[0]
        else:
            curPipe = lowerPipes[1]


        xdif = -playerx + curPipe["x"]
        ydif = -playery + curPipe["y"]

        return torch.tensor([xdif,ydif,vel],dtype=torch.float32)

    def dump_qvalues(self, force=False):
        """
        Dump the qvalues to the .pt file
        """
        if self.gameCNT % self.DUMPING_N == 0 or force:
            torch.save(self.policy_net.state_dict(),"data/qvalues.pt")
            print("Q-values updated on local file.")
