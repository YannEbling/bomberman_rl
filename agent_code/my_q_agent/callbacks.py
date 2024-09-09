import os
import pickle
import random
import time
import math

import numpy as np

from settings import *
from main import *
import auxiliary_functions as aux

#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

cols = (COLS - 2)
cells = pow((COLS - 2), 2)
n = COLS - 1
assert COLS == ROWS

RANDOM_ACTION = .15

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    #
    #   Time Analysis
    #

    
    #
    #   For Retraining Existing Model
    #
    if self.train and os.path.isfile("my-saved-model.pt"):
        #print("Continue training model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)
            #print("TYPE OF LOADED MODEL: ", type(self.Q))

    elif self.train or not os.path.isfile("my-saved-model.pt"):
    #if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #print("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        
        # ---
        
        # for one coin scenario only 
        #nr_states = pow(cells, 2)
        #self.Q = [[random.uniform(0.0, 1.0) for _ in range(5)] for _ in range(nr_states)]
        
        # ---

        # Setting up the model using the dimension reduction

        # First, compute the total number of disjoint states (after applying dimension reduction)

        number_of_agent_states = int(3 * n ** 2 / 16)  # for the agent, only a part of the tiles are valid positions
        number_of_coin_states = 1
        for k in range(1, n):
            if k % 2 == 0:
                number_of_coin_states += k // 2
            else:
                number_of_coin_states += k

        number_of_bomb_states = int((n - 1) ** 2 - (n / 2 - 1) ** 2) + 1  # this is the total number of valid positions
        # on the board plus one for the case of no bomb on the board
        nr_states = number_of_agent_states * number_of_bomb_states * number_of_coin_states
        shape = (nr_states, len(ACTIONS))
        self.logger.debug(f"New model has dimensions {shape}")
        self.Q = np.random.random(shape)
        
    else:
        self.logger.info("Loading model from saved state.")
        #print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)
            #print("TYPE OF LOADED MODEL: ", type(self.Q))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    #
    # to check, if q table is correctly saved and loaded
    #
    #for i in range(len(self.Q)):
    #    self.logger.debug(f"Q {self.Q[i]}")
    
    
    
    
    
    
    # todo Exploration vs exploitation
    #if self.train and random.random() < RANDOM_ACTION:
    if random.random() < RANDOM_ACTION:
        self.logger.debug("Choosing action purely at random.")
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0])

    self.logger.debug("Querying model for action.")
    

    
    #---
    # Get Q vector for current player position and coin position 
    # 'self': (str, int, bool, (int, int))


    agent_pos = game_state["self"][3]
    coin_index = aux.index_of_closest_item(agent_position=agent_pos, item_positions=game_state['coins'])
    bomb_positions = [bomb_attributes[0] for bomb_attributes in game_state['bombs']]
    bomb_index = aux.index_of_closest_item(agent_position=agent_pos, item_positions=bomb_positions)
    index, permutations = aux.state_to_index(game_state, coin_index=coin_index, bomb_index=bomb_index,
                                             dim_reduce=True, include_bombs=True)
    action_index = np.argmax(self.Q[index])
    permuted_action = ACTIONS[action_index]
    action_chosen = aux.revert_permutations(permuted_action, permutations)

    
    #if len(game_state["coins"]) > 0:
    #    coin_pos = game_state["coins"][0]
    #    coin_pos_x = coin_pos[0]
    #    coin_pos_y = coin_pos[1]
    #    coin_pos_index = (coin_pos_x - 1 + cols * (coin_pos_y - 1)) + 1
    
    #if coin_index is None:
    #    print("Couldn't find a coin")
    
    # 0 <= final_index <= 2400
    #final_index = agent_pos_index * (cells - 1) + coin_pos_index - 1
    actions = self.Q[action_index]
    #final_decision = np.argmax(actions)
    #action_chosen = ACTIONS[final_decision]

    self.logger.debug(f"Q[{index}]: {actions}")
    self.logger.debug(f"Argmax-action of row is {permuted_action}")
    self.logger.debug(f"Action chosen after reverting permutations {permutations}: {action_chosen}")
   
    #---
    return action_chosen


def find_closest_coin(game_state: dict):
    coins = game_state["coins"]
    agent = game_state["self"]
    
    if len(coins) <= 0:
        return None
        
    closest_coin = coins[0]
    closest_coin_dist = 1.7976931348623157e+308 # max float value
    for i in range(len(coins)):
        coin_pos = coins[i]
        agent_pos = agent[3]
        euclid_dist = math.sqrt(pow((coin_pos[0] - agent_pos[0]), 2) + pow((coin_pos[1] - agent_pos[1]), 2))
        if euclid_dist < closest_coin_dist:
            closest_coin = coins[i]
            closest_coin_dist = euclid_dist
    
    return closest_coin



def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
