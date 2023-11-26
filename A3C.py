import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size,):
        super(PolicyNetwork, self).__init__()
        #Network architecture
        # Shared layers
        hidden_size=256
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        # Policy-specific layers
        self.policy_fc = nn.Linear(hidden_size, output_size)

        # Value-specific layers
        self.value_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Shared layers
        x_shared = torch.relu(self.shared_fc1(x))
        x_shared = torch.relu(self.shared_fc2(x_shared))

        # Policy-specific layers
        logits = self.policy_fc(x_shared)
        policy_dist = Categorical(logits=logits)

        # Value-specific layers
        value = self.value_fc(x_shared)

        return policy_dist

class A3CAgent:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)

    def get_action(self, state):
        with torch.no_grad():
            action_prob = self.policy_network(torch.Tensor(state))
            action = action_prob.sample()
        return action.item()

    def train(self, state, action, advantage):
        # Forward pass to get the action probability distribution
        action_prob = self.policy_network(torch.Tensor(state))

        # Sample action from the distribution
        sampled_action = action_prob.sample()

        # Calculate log probability of the sampled action
        log_prob = action_prob.log_prob(sampled_action)

        # Compute policy loss and value loss
        policy_loss = -log_prob * torch.Tensor([advantage])

        # Use policy loss for both policy and value updates
        total_loss = policy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def map_index_to_uci(self, action_index, legal_moves,state):
    # Check if action_index is within a valid range
        if 0 <= action_index < len(legal_moves):
            return legal_moves[action_index]