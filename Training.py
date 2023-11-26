# Instantiate the chess game and agents
import chess
import torch
import numpy as np
from Board import ChessGame
from QlearnAgent import QLearningAgent
from A3C import A3CAgent
from A3C import PolicyNetwork
game = ChessGame()
state_size = 8 * 8 * 13
action_size=len(game.get_legal_moves())
q_agent = QLearningAgent(state_size, action_size)
value_estimate=PolicyNetwork(state_size, action_size)
a3c_agent = A3CAgent(value_estimate)
game_over=False
total_reward_q_agent=0
total_reward_a3c_agent=0
#check if the game is over
def is_terminal_state(board):
 # Define  criteria for a terminal state
    # Check for checkmate
    if board.is_checkmate():
        print("checkmate")
        return True, 
      

    # Check for stalemate
    if board.is_stalemate():
        print("stalemate")
        return True, 
        

    # Check for insufficient material
    if board.is_insufficient_material():
        print("Insufficient material")
        return True, 
        

    # Check for the seventy-five-move rule
    if board.is_seventyfive_moves():
        print("seventy-five-move rule")
        return True, 
        


#calculate advantage of the A3CAgent
def calculate_advantage(value_network, state, next_state, reward, gamma=0.99):
    # Calculate the discounted cumulative reward (return)
    discounted_return = reward + gamma * value_network(torch.Tensor(next_state)).sample().item()

    # Estimate the value of the current state
    estimated_value = value_network(torch.Tensor(state)).sample().item()

    # Advantage is the difference between the actual return and the estimated value
    advantage = discounted_return - estimated_value

    return advantage
#calculate reward for q agent
def calculate_reward(board):
    # Check if the game is over
    if board.is_checkmate():
        # If the game is checkmate, give a high positive reward
        return 10.0 if board.turn == chess.WHITE else -10.0
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        # If the game is a draw or there is insufficient material, give a neutral reward
        return 0.0
    else:
        # For non-terminal states,
        # For example, give a reward proportional to the difference in material counts
        white_material = sum(value.piece_type for square, value in board.piece_map().items() if value.color == chess.WHITE)
        black_material = sum(value.piece_type for square, value in board.piece_map().items() if value.color == chess.BLACK)
        return float(white_material - black_material)
#get board state
def get_current_state(board):
    # Create a 8x8x13 matrix to represent the board
    # 6 channels for each piece type (pawn, knight, bishop, rook, queen, king)
    # 2 channels for each color (white, black)
    # 1 channel for indicating the current player's turn 0 for black's turn and 1 for white's turn
    state = np.zeros((8, 8, 13), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))  # Note: Invert the row index
            if piece is not None:
                # Encode piece type
                piece_type = piece.piece_type - 1  # piece_type ranges from 1 to 6, convert to 0 to 5
                channel = piece_type

                # Encode piece color
                if piece.color == chess.WHITE:
                    channel += 6  # Offset for white pieces

                state[row, col, channel] = 1

    # Add a channel to indicate the current player's turn
    state[:, :, 12] = int(board.turn)

    return state.flatten()
num_episodes=1000;
# Training loop
for episode in range(num_episodes):
    game.reset()
    state = get_current_state(game.board)  # =a function to convert the board state to a format suitable for agents
    while not game_over:
        # Q-learning agent makes a move
        q_action = q_agent.get_action(state)
        while  q_action>len(game.get_legal_moves())-1:
            q_action = q_agent.get_action(state)
        uci_action=q_agent.map_index_to_uci(q_action,game.get_legal_moves())
        game.make_move(uci_action)
        next_state = get_current_state(game.board)
        reward = calculate_reward(game.board)
        total_reward_q_agent=total_reward_q_agent+reward
        q_agent.update_q_value(state, q_action, reward,next_state)
        print("After Qlearning agent plays the board is")
        print(game.board)
        game_over=is_terminal_state(game.board)
        if game_over:
            print("QLearning agent wins")
            print("Total reward for A3C agent "+str(total_reward_a3c_agent))
            print("Total reward for Q-learning agent "+str(total_reward_q_agent))
            break
        # A3C agent makes a move
        a3c_action = a3c_agent.get_action(state)
        while a3c_action>len(game.get_legal_moves())-1:
            a3c_action = a3c_agent.get_action(state)
        uci_action=a3c_agent.map_index_to_uci(a3c_action,game.get_legal_moves(),state)
        game.make_move(uci_action)
        next_state = get_current_state(game.board)
        reward = calculate_reward(game.board)
        total_reward_a3c_agent=total_reward_a3c_agent+reward
        advantage = calculate_advantage(value_estimate,state,next_state,reward)
        a3c_agent.train(state, a3c_action, advantage)
        print("After A3C agent plays the board is")
        print(game.board)
        game_over=is_terminal_state(game.board)
        if game_over:
            print("A3C agent wins")
            print("Total reward for A3C agent "+str(total_reward_a3c_agent))
            print("Total reward for Q-learning agent "+str(total_reward_q_agent))
            break
