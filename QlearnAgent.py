import numpy as np
class QLearningAgent:
    def __init__(self, state_size, action_size, epsilon=0.1,learning_rate=0.1, discount_factor=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))
    def map_index_to_uci(self, action_index, legal_moves):
    # Check if action_index is within a valid range
        if 0 <= action_index < len(legal_moves):
            return legal_moves[action_index]
        
    def get_action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            action = np.random.choice(self.action_size)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            action = np.argmax(self.q_table[state])
            # Convert the action index to UCI notation in map_index_to_uci
        return action
    
    def update_q_value(self, state, action, reward, next_state):
        # Update Q-value based on the Q-learning update rule
        current_q_value = self.q_table[state, action]
        max_next_q_value = np.max(self.q_table[next_state, :])
        
        # Q-learning update rule
        updated_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)

        # Update the Q-table
        self.q_table[state, action] = updated_q_value