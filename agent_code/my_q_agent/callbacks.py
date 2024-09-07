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
ESCAPE_ROUTE_STATES = 16    # up, right, down, left


RANDOM_ACTION = .1
RANDOM_ACTION_LOW = .05

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
        
        nr_states = pow(cells, 3) + ESCAPE_ROUTE_STATES
        self.Q = np.random.rand(nr_states, len(ACTIONS)).astype(np.float32)
        
        
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

    
    #---
    # Get Q vector for current player position and coin position 
    # 'self': (str, int, bool, (int, int))
    agent_pos = game_state["self"][3]
    agent_pos_index = compute_agent_pos_index(agent_pos)

    # coin, crate, bomb positions and indices
    closest_coin = find_closest_coin(game_state)
    closest_crate = find_closest_crate(game_state)
    closest_bomb = find_closest_bomb(game_state)
    coin_pos_index, crate_pos_index, bomb_pos_index = compute_indices(closest_coin, closest_crate, closest_bomb)





    # todo Exploration vs exploitation
    if self.train and random.random() < RANDOM_ACTION:
    #if random.random() < RANDOM_ACTION_LOW:
        self.logger.debug("Choosing action purely at random.")
        #return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .0, .1])
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .0, .2])
        
    # only random action during normal game, when no bomb is close enough
    elif random.random() < RANDOM_ACTION:
        if closest_bomb == None:
            print("random act")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .0, .2])
        elif dist(agent_pos[0], closest_bomb[0][0], agent_pos[1], closest_bomb[0][1]) >= 4.0:
            print("random act")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .0, .2])
        
            
                

    self.logger.debug("Querying model for action.")



    #
    #   Decision Making
    #
    
    pull_index = compute_pull_factor_index(game_state, crate_pos_index, coin_pos_index)
        
    
    
    
    #
    #   Compute Index
    #
    bomb_under_feet = is_bomb_under_players_feet(game_state)
    state_index = compute_state_index(game_state, agent_pos_index, pull_index, bomb_pos_index, bomb_under_feet)
    actions = self.Q[state_index]
    final_decision = np.argmax(actions)
    action_chosen = ACTIONS[final_decision]
    
    #
    #   Logging
    #
    self.logger.debug(f"Action choosen: {action_chosen}")
    self.logger.debug(f"Q[{state_index}]: {actions}")
   
    return action_chosen

def dist(x0, x1, y0, y1):
    return math.sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2))

def can_escape_left(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]
    
    field = game_state["field"]
    
    for i in range(1, 4):
        
        # check for out of bounds
        if ag_x - i < 0 or ag_x - i >= ROWS:
            return 0
        
        # check infront
        tile = field[ag_x - i][ag_y]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0
            
        # check top
        tile = field[ag_x - i][ag_y - 1]
        if tile == 0: 
            return 1
        
        # check bottom
        tile = field[ag_x - i][ag_y + 1]
        if tile == 0:
            return 1
    
    # check for out of bounds
    if ag_x - 4 < 0 or ag_x - 4 >= ROWS:
        return 0
        
    # check last tile that is right out of range of bomb
    tile = field[ag_x - 4][ag_y]
    if tile == 0:
        return 1
        
    return 0

def can_escape_right(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]
    
    field = game_state["field"]
    
    for i in range(1, 4):
    
        # check for out of bounds
        if ag_x + i < 0 or ag_x + i >= ROWS:
            return 0
    
        # check infront
        tile = field[ag_x + i][ag_y]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0
            
        # check top
        tile = field[ag_x + i][ag_y - 1]
        if tile == 0: 
            return 1
        
        # check bottom
        tile = field[ag_x + i][ag_y + 1]
        if tile == 0:
            return 1
    
    # check for out of bounds
    if ag_x + 4 < 0 or ag_x + 4 >= ROWS:
        return 0
    
    # check last tile that is right out of range of bomb
    tile = field[ag_x + 4][ag_y]
    if tile == 0:
        return 1
        
    return 0
    
def can_escape_up(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]
    
    field = game_state["field"]
    
    for i in range(1, 4):
    
        # check for out of bounds
        if ag_y - i < 0 or ag_y - i >= COLS:
            return 0
    
        # check infront
        tile = field[ag_x][ag_y - i]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0
            
        # check left
        tile = field[ag_x - 1][ag_y - i]
        if tile == 0: 
            return 1
        
        # check right
        tile = field[ag_x + 1][ag_y - i]
        if tile == 0:
            return 1
            
    # check for out of bounds
    if ag_y - 4 < 0 or ag_y - 4 >= ROWS:
        return 0
    
    # check last tile that is right out of range of bomb
    tile = field[ag_x][ag_y - 4]
    if tile == 0:
        return 1
        
    return 0
    
def can_escape_down(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]
    
    field = game_state["field"]
    
    for i in range(1, 4):
    
        # check for out of bounds
        if ag_y + i < 0 or ag_y + i >= COLS:
            return 0
    
        # check infront
        tile = field[ag_x][ag_y + i]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0
            
        # check left
        tile = field[ag_x - 1][ag_y + i]
        if tile == 0: 
            return 1
        
        # check right
        tile = field[ag_x + 1][ag_y + i]
        if tile == 0:
            return 1
    
    # check for out of bounds
    if ag_y + 4 < 0 or ag_y + 4 >= ROWS:
        return 0
    
    # check last tile that is right out of range of bomb
    tile = field[ag_x][ag_y + 4]
    if tile == 0:
        return 1
        
    return 0

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

def is_bomb_under_players_feet(game_state):
    agent_pos = game_state["self"][3]
    bombs = game_state["bombs"]
    if bombs == None:
        return False
    for bomb in bombs:
        bomb_pos = bomb[0]
        if agent_pos[0] == bomb_pos[0] and agent_pos[1] == bomb_pos[1]:
            return True
    return False

def compute_state_index(game_state, agent_pos_index, pull_index, bomb_pos_index, bomb_under_players_feet):
    if bomb_under_players_feet:
        left = can_escape_left(game_state)
        right = can_escape_right(game_state)
        up = can_escape_up(game_state)
        down = can_escape_down(game_state)
        index = left * 8 + right * 4 + up * 2 + down * 1
        return index
    
    index = agent_pos_index * (cells - 1) + pull_index * (cols - 1) + bomb_pos_index - 1      # is this correct?
    index = index + 16 # for the new 16 bomb dropped, available escape routes states
    
    return index

def compute_agent_pos_index(agent_pos): 
    return (agent_pos[0] - 1 + cols * (agent_pos[1] - 1)) + 1    

def compute_pull_factor_index(game_state, crate_pos_index, coin_pos_index):
    
    closest_coin = find_closest_coin(game_state)
    closest_crate = find_closest_crate(game_state)
    closest_bomb = find_closest_bomb(game_state)
    
    pull_index = 1
    if closest_coin == None and closest_crate != None:  # only crate, use crate index
        pull_index = crate_pos_index
    elif closest_coin != None and closest_crate == None: # only coin, use coin index
        pull_index = coin_pos_index
    elif closest_coin != None and closest_crate != None: # crate and coin, prefer coin over crate
        pull_index = coin_pos_index
    return pull_index

def compute_indices(closest_coin, closest_crate, closest_bomb):

    #   Coin
    coin_pos_index = 1  # this could be an issue
    if closest_coin != None:
        coin_pos_index = (closest_coin[0] - 1 + cols * (closest_coin[1] - 1)) + 1

    #   Crate
    crate_pos_index = 1  # this could be an issue
    if closest_crate != None:
        crate_pos_index = (closest_crate[0] - 1 + cols * (closest_crate[1] - 1)) + 1

    #   Bomb
    bomb_pos_index = 0  # this could be an issue
    if closest_bomb != None:
        bomb_pos_index = (closest_bomb[0][0] - 1 + cols * (closest_bomb[0][1] - 1)) + 1
    else:
        pass
        # todo: what if there is no bomb?
      
    return coin_pos_index, crate_pos_index, bomb_pos_index
   

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
