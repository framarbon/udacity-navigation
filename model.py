import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_layers = [state_size, 32, 128, 521]
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)
#         self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
#             x = self.dropout(x)
        return self.output(x)
