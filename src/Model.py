
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.nn = nn.Sequential(nn.Linear(4, 64), 
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64), 
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 1))

        self.softmax = nn.Softmax(dim = 0)
        self.saved_actions = []
        self.rewards = []
        

    def forward(self, x):

        x = self.nn(x)
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = self.softmax(x[:-1, 0])
        
        return action_prob
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.nn = nn.Sequential(nn.Linear(4, 64), 
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64), 
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 1))
        
    def forward(self, x):
        #input layer
        x = self.nn(x)
        #get state value
        # critic: evaluates being in the state s_t
        value = x[-1, 0]
        
        return value
