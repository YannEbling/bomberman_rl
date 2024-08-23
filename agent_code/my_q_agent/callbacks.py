import os
import pickle
import random

import numpy as np


#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


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
    #   For Retraining Existing Model
    #
    #if self.train and os.path.isfile("my-saved-model.pt"):
    #    print("Continue training model from saved state.")
    #    with open("my-saved-model.pt", "rb") as file:
    #        self.Q = pickle.load(file)        
    #elif self.train or not os.path.isfile("my-saved-model.pt"):
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        
        # ---
        
        # for one coin scenario only 
        # Initialize
        # For 7x7 agent position in arena states and 7x7 coin position in arena (= 2401 states) containing the 5 chances for the 5 actions
        self.Q = [[random.uniform(0.0, 1.0) for _ in range(5)] for _ in range(2401)]
        
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
    random_prob = .15
    if self.train and random.random() < random_prob:
    #if random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0])

    self.logger.debug("Querying model for action.")
    

    
    #---
    # Get Q vector for current player position and coin position 
    # 'self': (str, int, bool, (int, int))
    
    
    # (1,1) <= agent_position <= (7,7) = 1 <= index <= 49
    agent_pos = game_state["self"][3]
    agent_pos_x = agent_pos[0]
    agent_pos_y = agent_pos[1]
    agent_pos_index = (agent_pos_x - 1 + 7 * (agent_pos_y - 1)) + 1
    
    
    
    coin_pos_index = 1
    
    if len(game_state["coins"]) > 0:
        coin_pos = game_state["coins"][0]
        coin_pos_x = coin_pos[0]
        coin_pos_y = coin_pos[1]
        coin_pos_index = (coin_pos_x - 1 + 7 * (coin_pos_y - 1)) + 1
    
    # 0 <= final_index <= 2400
    final_index = agent_pos_index * 48 + coin_pos_index - 1
    actions = self.Q[final_index]
    final_decision = np.argmax(actions)
    action_chosen = ACTIONS[final_decision]
    
    self.logger.debug(f"Action choosen: {action_chosen}")
    self.logger.debug(f"Q[{final_index}]: {actions}")
   
    #---
    
    return  action_chosen


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
