import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()
        self.softmax = nn.Softmax(dim = 0)
        self.saved_actions = []
        self.rewards = []
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = self.softmax(x[:-1, 0])
        # critic: evaluates being in the state s_t
        value = x[-1, 0]
        return action_prob, value
