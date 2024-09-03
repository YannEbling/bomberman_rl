from collections import namedtuple, deque

import pickle
from typing import List

import math

import events as e
from .callbacks import state_to_features
from .callbacks import find_closest_coin
from .callbacks import find_closest_bomb
from .callbacks import find_closest_crate

import numpy as np
from main import *

import time

from settings import *

import os

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
#PLACEHOLDER_EVENT = "PLACEHOLDER"

# Custom

# Learning rate alpha, 0.0 < alpha < 1.0
ALPHA = 0.4


# Discount factor gamma, 0.0 < gamma < 1.0
GAMMA = 0.2


MyTransition = namedtuple('Transition', ('state', 'action'))
MY_TRANSITION_HISTORY_SIZE = 40


# Action to index
action_to_index = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "WAIT": 4,
    "BOMB": 5
}

# For training debugging
coins_collected = 0
bombs_survived = 0
round = 0

id = os.getpid()

# convert setting values to used values in training
cols = (COLS - 2)
cells = pow((COLS - 2), 2)

# Execution Time Analysis

# Saving data every round is costly, save only every n rounds
SAVE_INTERVAL = 1000
save_counter = 0

active_bombs = []

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    
    
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    #---
    self.mytransitions = deque(maxlen=MY_TRANSITION_HISTORY_SIZE)
    



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Idea: Add your own events to hand out rewards
    #if ...:
        #events.append(PLACEHOLDER_EVENT)










    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    

    #
    #
    #   ---   Compute State Index of old_game_state 
    #
    #
    
    
    #
    #   Agent
    #
    old_agent_pos = old_game_state["self"][3]
    old_agent_pos_index = (old_agent_pos[0] - 1 + cols * (old_agent_pos[1] - 1)) + 1
        

    
        
        
    #
    #   Coin
    #
    old_closest_coin = find_closest_coin(old_game_state)
    old_coin_pos_index = 1  # this could be an issue
    if old_closest_coin != None:
        old_coin_pos_index = (old_closest_coin[0] - 1 + cols * (old_closest_coin[1] - 1)) + 1


    #
    #   Crate
    #
    old_closest_crate = find_closest_crate(old_game_state)
    old_crate_pos_index = 1  # this could be an issue
    if old_closest_crate != None:
        old_crate_pos_index = (old_closest_crate[0] - 1 + cols * (old_closest_crate[0] - 1)) + 1


    #
    #   Bomb
    #
    old_closest_bomb = find_closest_bomb(old_game_state)
    old_bomb_pos_index = 1  # this could be an issue
    if old_closest_bomb != None:
        old_bomb_pos_index = (old_closest_bomb[0][0] - 1 + cols * (old_closest_bomb[0][1] - 1)) + 1
    else:
        pass
        # todo: what if there is no bomb?
        
    #
    #   Compute Pull Factor
    #
    old_pull_index = 1
    if old_closest_coin == None and old_closest_crate != None:  # only crate, use crate index
        old_pull_index = old_crate_pos_index
    elif old_closest_coin != None and old_closest_crate == None: # only coin, use coin index
        old_pull_index = old_coin_pos_index
    elif old_closest_coin != None and old_closest_crate != None: # crate and coin, prefer coin over crate
        old_pull_index = old_coin_pos_index
        
    
 
        
    #   Compute Index
    #old_state_index = old_agent_pos_index * (cells - 1) + old_coin_pos_index - 1
    old_state_index = old_agent_pos_index * (cells - 1) + old_pull_index * (cols - 1) + old_bomb_pos_index - 1      # is this correct?

    #   Compute Action Index of old_game_state
    old_action_index = action_to_index[self_action]
    
    
    
    
    #
    #   Compute State Index of new_game_state
    #
    # (1,1) <= agent_position <= (7,7) = 1 <= index <= 49
    #new_agent_pos   = new_game_state["self"][3]
    #new_agent_pos_x = new_agent_pos[0]
    #new_agent_pos_y = new_agent_pos[1]
    #new_agent_pos_index = (new_agent_pos_x - 1 + cols  * (new_agent_pos_y - 1)) + 1
    
    # Coin position
    #new_possible_closest_coin = find_closest_coin(new_game_state)
    #new_coin_pos_index = 1
    
    #if new_possible_closest_coin != None:
    #    new_coin_pos_index = (new_possible_closest_coin[0] - 1 + cols * (new_possible_closest_coin[1] - 1)) + 1
    #else:
        #print("Couldnt find a coin")
    
    
    #
    #
    #   ---   Compute State Index of new_game_state
    #
    #
    
    #
    #   Agent
    #
    new_agent_pos = new_game_state["self"][3]
    new_agent_pos_index = (new_agent_pos[0] - 1 + cols * (new_agent_pos[1] - 1)) + 1
    
    
        
        
    #
    #   Coin
    #
    new_closest_coin = find_closest_coin(new_game_state)
    new_coin_pos_index = 1  # this could be an issue
    if new_closest_coin != None:
        new_coin_pos_index = (new_closest_coin[0] - 1 + cols * (new_closest_coin[1] - 1)) + 1


    #
    #   Crate
    #
    new_closest_crate = find_closest_crate(new_game_state)
    new_crate_pos_index = 1  # this could be an issue
    if new_closest_crate != None:
        new_crate_pos_index = (new_closest_crate[0] - 1 + cols * (new_closest_crate[0] - 1)) + 1


    #
    #   Bomb
    #
    new_closest_bomb = find_closest_bomb(new_game_state)
    new_bomb_pos_index = 1  # this could be an issue
    if new_closest_bomb != None:
        new_bomb_pos_index = (new_closest_bomb[0][0] - 1 + cols * (new_closest_bomb[0][1] - 1)) + 1
    else:
        pass
        # todo: what if there is no bomb?


    #
    #   Custom Event: Survived a bomb -> SURVIVED_BOMB
    #

    # Add current bombs on the field, that are not already in the active bombs list, to the list
    # But add only the position tupel, for simplification
    for i in old_game_state["bombs"]:
        if i[0] not in active_bombs:
            active_bombs.append(i[0])

    # Make a temporary list of the position tuples of all bombs currently on the field
    tmp = []
    for i in old_game_state["bombs"]:
        tmp.append(i[0])

    # Check for all active bombs, if they are not in the temporary list
    # This can only be the case, when a bomb exploded
    # If the agent is still alive at this point, give out reward
    for i in active_bombs:
        if i not in tmp:
            print("survived a bomb!")
            active_bombs.remove(i)
            events.append(e.SURVIVED_BOMB)
            global bombs_survived
            bombs_survived = bombs_survived + 1

    #
    #   Custom event: Stepped away from bomb
    #

    if old_closest_bomb != None:
        print(f"old_agent_pos: {old_agent_pos}")
        print(f"new_agent_pos: {new_agent_pos}")

        old_dist_to_bomb = math.sqrt(pow((old_closest_bomb[0][0] - old_agent_pos[0]), 2) + pow((old_closest_bomb[0][1] - old_agent_pos[1]), 2))
        new_dist_to_bomb = math.sqrt(pow((old_closest_bomb[0][0] - new_agent_pos[0]), 2) + pow((old_closest_bomb[0][1] - new_agent_pos[1]), 2))

        print(f"old_dist_to_bomb: {old_dist_to_bomb}")
        print(f"new_dist_to_bomb: {new_dist_to_bomb}")

        if new_dist_to_bomb > old_dist_to_bomb:
            print("stepped away from bomb")
            events.append(e.STEPPED_AWAY_FROM_BOMB)



    #
    #   Custom Event: Too close to bomb -> IN_DANGER
    #

    #if old_closest_bomb != None:

    #    dist_to_bomb = math.sqrt(pow((old_closest_bomb[0][0] - old_agent_pos[0]), 2) + pow((old_closest_bomb[0][1] - old_agent_pos[1]), 2))

    #    x_aligned = False
    #    if old_closest_bomb[0][0] == old_agent_pos[0]:
    #        x_aligned = True

    #    y_aligned = False
    #    if old_closest_bomb[0][1] == old_agent_pos[1]:
    #        y_aligned = True
#
    #    if (x_aligned or y_aligned) and dist_to_bomb <= 3:
    #        events.append(e.IN_DANGER)
    #        print("IN DANGER")
        
    #
    #   Compute Pull Factor
    #
    new_pull_index = 1
    if new_closest_coin == None and new_closest_crate != None:  # only crate, use crate index
        new_pull_index = new_crate_pos_index
    elif new_closest_coin != None and new_closest_crate == None: # only coin, use coin index
        new_pull_index = new_coin_pos_index
    elif new_closest_coin != None and new_closest_crate != None: # crate and coin, prefer coin over crate
        new_pull_index = new_coin_pos_index
        
          
        
    
    
    #   Compute Index
    #new_state_index = new_agent_pos_index * (cells - 1) + new_coin_pos_index - 1
    new_state_index = new_agent_pos_index * (cells - 1) + new_pull_index * (cols - 1) + new_bomb_pos_index - 1      # is this correct?










    #
    #   Update Q-value of state-action tupel
    #
    reward = reward_from_events(self, events)
    argmax = np.argmax( self.Q[new_state_index] )
    
    factor1 = ( 1.0 - ALPHA ) * self.Q[old_state_index][old_action_index]
    factor2 = GAMMA * self.Q[new_state_index][argmax]
    factor3 = ALPHA * ( reward + factor2 )
    #factor2 = ALPHA * reward
    #factor3 = ALPHA * GAMMA * np.argmax( self.Q[new_state_index] )
    
    #
    #   Logging
    #
    #self.logger.debug(f"np.argmax: {np.argmax(self.Q[new_state_index])}")
    #self.logger.debug(f"np.argmax value: {self.Q[new_state_index][argmax]}")
    #self.logger.debug(f"factor1 :{factor1}")
    #self.logger.debug(f"factor2 :{factor2}")
    #self.logger.debug(f"factor3 :{factor3}")
    #self.logger.debug(f"old q value before update: {self.Q[old_state_index][old_action_index]}")
    
    new_value = factor1 + factor2 + factor3
    #self.logger.debug(f"new q value: {new_value}")
    #self.logger.debug(f"reward: {reward}")
    
    # set new value
    self.Q[old_state_index][old_action_index] = new_value
    
    #self.logger.debug(f"new q value before update: {self.Q[old_state_index][old_action_index]}")
    
    self.mytransitions.append(MyTransition(new_game_state, self_action))
    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.
    
    :param self: The same object that is passed to all of your callbacks.
    """


    #
    #   Custom event survived bomb cleanup
    #
    active_bombs.clear()


    #
    #
    #   ---   Compute State Index of last_game_state
    #
    #
    
    #
    #   Agent
    #
    last_agent_pos = last_game_state["self"][3]
    last_agent_pos_index = (last_agent_pos[0] - 1 + cols * (last_agent_pos[1] - 1)) + 1
    
    
        
        
    #
    #   Coin
    #
    last_closest_coin = find_closest_coin(last_game_state)
    last_coin_pos_index = 1  # this could be an issue
    if last_closest_coin != None:
        last_coin_pos_index = (last_closest_coin[0] - 1 + cols * (last_closest_coin[1] - 1)) + 1


    #
    #   Crate
    #
    last_closest_crate = find_closest_crate(last_game_state)
    last_crate_pos_index = 1  # this could be an issue
    if last_closest_crate != None:
        last_crate_pos_index = (last_closest_crate[0] - 1 + cols * (last_closest_crate[0] - 1)) + 1


    #
    #   Bomb
    #
    last_closest_bomb = find_closest_bomb(last_game_state)
    last_bomb_pos_index = 1  # this could be an issue
    if last_closest_bomb != None:
        last_bomb_pos_index = (last_closest_bomb[0][0] - 1 + cols * (last_closest_bomb[0][1] - 1)) + 1
    else:
        pass
        # todo: what if there is no bomb?
        
        
        
    #
    #   Compute Pull Factor
    #
    last_pull_index = 1
    if last_closest_coin == None and last_closest_crate != None:  # only crate, use crate index
        last_pull_index = last_crate_pos_index
    elif last_closest_coin != None and last_closest_crate == None: # only coin, use coin index
        last_pull_index = last_coin_pos_index
    elif last_closest_coin != None and last_closest_crate != None: # crate and coin, prefer coin over crate
        last_pull_index = last_coin_pos_index
        
          
        
    
    
    #   Compute Index
    #last_state_index = last_agent_pos_index * (cells - 1) + last_coin_pos_index - 1
    last_state_index = last_agent_pos_index * (cells - 1) + last_pull_index * (cols - 1) + last_bomb_pos_index - 1      # is this correct?


 
    last_action_index = action_to_index[last_action]

    #
    #   Update Q-value of state-action tupel
    #
    reward = reward_from_events(self, events)
    argmax = np.argmax( self.Q[last_state_index] )

    factor1 = ( 1.0 - ALPHA ) * self.Q[last_state_index][last_action_index]
    factor2 = GAMMA * self.Q[last_state_index][argmax]
    factor3 = ALPHA * ( reward + factor2 )
    new_value = factor1 + factor2 + factor3
    self.Q[last_state_index][last_action_index] = new_value

    print(f"reward: {reward}")

    #   Debug - coin found?
    for event in events:
        if event == e.COIN_COLLECTED:
            global coins_collected 
            coins_collected = coins_collected + 1
        if event == e.SURVIVED_BOMB:
            global bombs_survived
            bombs_survived = bombs_survived + 1
    
    print(f"Coins collected: {coins_collected}")
    print(f"Bombs survived: {bombs_survived}")


    #for i in range(len(self.Q)):
    #    self.logger.debug(self.Q[i])

    global round
    round = round + 1
    self.logger.debug(f"Round nr: {round}")

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    training_time = time.time()

    #
    # Store the model
    #
    global save_counter
    save_counter = save_counter + 1
    print(f"save counter: {save_counter}")
    if save_counter == SAVE_INTERVAL:
        print("saving data")
        if os.path.isfile("./mp/mp.hky"):
            with open(f"mp/data/my-saved-model{id}.pt", "wb") as file:
                pickle.dump(self.Q, file)
                file.close()
        else:
            with open("my-saved-model.pt", "wb") as file:
                pickle.dump(self.Q, file)
                file.close()

        save_counter = 0


    #
    #   Execution Time Analysis
    #
    exec_time = time.time() - training_time
    print(f"Execution time for function \"end of round\": {exec_time:.6f} seconds")
    self.logger.debug(f"Execution time for function \"end of round\": {exec_time} seconds")
 

def training_end():
    pass
    
   

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    # coin collected?

    
    
    game_rewards = {
        #e.COIN_COLLECTED: 15,
        #e.KILLED_OPPONENT: 5,
        #e.WAITED: -0.2,
        #e.INVALID_ACTION: -3.2,
        #e.MOVED_RIGHT: 0.2,
        #e.MOVED_UP: 0.2,
        #e.MOVED_DOWN: 0.2,
        #e.MOVED_LEFT: 0.2,
        #e.BOMB_DROPPED: 0.0,
        #e.CRATE_DESTROYED: 10.0,
        #e.IN_DANGER: -5,
        #e.KILLED_SELF: -5.0,
        e.STEPPED_AWAY_FROM_BOMB: 1.0,
        e.SURVIVED_BOMB: 10.0
        #PLACEHOLDER_EVENT: -.05  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            if event == e.SURVIVED_BOMB:
                print("bomb survied lol")
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
