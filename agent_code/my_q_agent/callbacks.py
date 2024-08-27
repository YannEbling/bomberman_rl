import os
import pickle
import random
from time import sleep, time
import math

import numpy as np

from settings import *
from main import *

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

cols = (COLS - 2)
cells = pow((COLS - 2), 2)

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
        print("Continue training model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)        
    elif self.train or not os.path.isfile("my-saved-model.pt"):
    #if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        
        # ---
        
        nr_states = pow(cells, 3)
        print(nr_states)
        sleep(5)
        # Define the length of the array
        array_length = 11390625 * 6

        # Create a NumPy array with random values between 0 and 1 and dtype of float32
        random_array = np.random.rand(array_length).astype(np.float32)
        print(len(random_array))
        print(random_array.shape)
        for i in random_array:
            print(i)
        
        print(random_array)
        sleep(10)
        
        #self.Q = [[random.uniform(0.0, 1.0) for _ in range(6)] for _ in range(nr_states)]
        #print(len(self.Q))
        
        # ---
        
    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)


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
        #return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0])
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .0, .1])

    self.logger.debug("Querying model for action.")
    

    
    #---
    # Get Q vector for current player position and coin position 
    # 'self': (str, int, bool, (int, int))
    
    
    agent_pos = game_state["self"][3]
    agent_pos_x = agent_pos[0]
    agent_pos_y = agent_pos[1]
    agent_pos_index = (agent_pos_x - 1 + cols * (agent_pos_y - 1)) + 1
    
    
    
    
    
    
    #
    #   Coin
    #
    closest_coin = find_closest_coin(game_state)
    coin_pos_index = 1  # this could be an issue
    #print(closest_coin)
    
    if closest_coin != None:
        coin_pos_index = (closest_coin[0] - 1 + cols * (closest_coin[1] - 1)) + 1

    #
    #   Crate
    #
    closest_crate = find_closest_crate(game_state)
    crate_pos_index = 1  # this could be an issue
    #print(closest_crate)
    
    if closest_crate != None:
        crate_pos_index = (closest_crate[0] - 1 + cols * (closest_crate[0] - 1)) + 1
    
    
    #
    #   Bomb
    #
    closest_bomb = find_closest_bomb(game_state)
    bomb_pos_index = 1  # this could be an issue
    print(closest_bomb)
    
    if closest_bomb != None:
        bomb_pos_index = (closest_bomb[0][0] - 1 + cols * (closest_bomb[0][1] - 1)) + 1


    #
    #   Decision Making
    #
    # 0 <= final_index <= 2400
    
    
    
    
    
    
    
    final_index = agent_pos_index * (cells - 1) + coin_pos_index - 1
    
    
    actions = self.Q[final_index]
    final_decision = np.argmax(actions)
    action_chosen = ACTIONS[final_decision]
    
    #
    #   Logging
    #
    self.logger.debug(f"Action choosen: {action_chosen}")
    self.logger.debug(f"Q[{final_index}]: {actions}")
   
    
   
   
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

def find_closest_bomb(game_state: dict):
    bombs = game_state["bombs"]
    agent = game_state["self"]
    
    if len(bombs) <= 0:
        return None
        
    closest_bomb = bombs[0]
    closest_bomb_dist = 1.7976931348623157e+308 # max float value
    for i in range(len(bombs)):
        bomb_pos = bombs[i][0]
        agent_pos = agent[3]
        euclid_dist = math.sqrt(pow((bomb_pos[0] - agent_pos[0]), 2) + pow((bomb_pos[1] - agent_pos[1]), 2))
        if euclid_dist < closest_bomb_dist:
            closest_bomb = bombs[i]
            closest_bomb_dist = euclid_dist
    
    return closest_bomb
    
def find_closest_crate(game_state: dict):
    field = game_state["field"]
    agent = game_state["self"]
    field_shape = field.shape
    
    closest_crate = None
    closest_crate_dist = 1.7976931348623157e+308 # max float value
    for x in range(field_shape[0]): # is this x starting at top left?
        for y in range(field_shape[1]): # is this y starting at top left?
            tile = field[x][y]  
            if tile == 1: # 1 = crate
                agent_pos = agent[3]
                euclid_dist = math.sqrt(pow((x - agent_pos[0]), 2) + pow((y - agent_pos[1]), 2))
                if euclid_dist < closest_crate_dist:
                    closest_crate = (x, y)
                    closest_crate_dist = euclid_dist
    
    return closest_crate


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
