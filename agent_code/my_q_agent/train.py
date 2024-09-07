from collections import namedtuple, deque

import pickle
from typing import List

import math

import events as e
from .callbacks import state_to_features
from .callbacks import find_closest_coin
from .callbacks import find_closest_bomb
from .callbacks import find_closest_crate
from .callbacks import compute_state_index
from .callbacks import compute_agent_pos_index
from .callbacks import compute_pull_factor_index
from .callbacks import compute_indices
from .callbacks import can_escape_left
from .callbacks import can_escape_right
from .callbacks import can_escape_up
from .callbacks import can_escape_down
from .callbacks import is_bomb_under_players_feet


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
    #   Agent
    old_agent_pos = old_game_state["self"][3]
    old_agent_pos_index = compute_agent_pos_index(old_agent_pos)
          
    #   Coin, Crate, Bomb 
    old_closest_coin = find_closest_coin(old_game_state)
    old_closest_crate = find_closest_crate(old_game_state)
    old_closest_bomb = find_closest_bomb(old_game_state)    
    old_coin_pos_index, old_crate_pos_index, old_bomb_pos_index = compute_indices(old_closest_coin, old_closest_crate, old_closest_bomb)
    old_is_bomb_under_players_feet = is_bomb_under_players_feet(old_game_state)
        
    #   Compute Pull Factor
    old_pull_index = compute_pull_factor_index(old_game_state, old_crate_pos_index, old_coin_pos_index)
            
    #   Compute Index
    old_state_index = compute_state_index(old_game_state, old_agent_pos_index, old_pull_index, old_bomb_pos_index, old_is_bomb_under_players_feet)

    #   Compute Action Index of old_game_state
    old_action_index = action_to_index[self_action]
    
    

    
    #
    #
    #   ---   Compute State Index of new_game_state
    #
    #
    #   Agent
    new_agent_pos = new_game_state["self"][3]
    new_agent_pos_index = compute_agent_pos_index(new_agent_pos)
        
    #   Coin, Crate, Bomb
    new_closest_coin = find_closest_coin(new_game_state)
    new_closest_crate = find_closest_crate(new_game_state)
    new_closest_bomb = find_closest_bomb(new_game_state)    
    new_coin_pos_index, new_crate_pos_index, new_bomb_pos_index = compute_indices(new_closest_coin, new_closest_crate, new_closest_bomb)
    new_is_bomb_under_players_feet = is_bomb_under_players_feet(new_game_state)

    #   Compute Pull Factor
    new_pull_index = compute_pull_factor_index(new_game_state, new_crate_pos_index, new_coin_pos_index)
        
    #   Compute Index
    new_state_index = compute_state_index(new_game_state, new_agent_pos_index, new_pull_index, new_bomb_pos_index, new_is_bomb_under_players_feet)




    #
    #   Custom Event: Bomb placed directly at crate?
    #

    for event in events:
        if event == e.BOMB_DROPPED:             
            dist_to_crate = math.sqrt(pow((new_closest_crate[0] - new_agent_pos[0]), 2) + pow((new_closest_crate[1] - new_agent_pos[1]), 2))
            if dist_to_crate <= 1.01:
                events.append(e.BOMB_DROPPED_NEXT_TO_CRATE)
                print("BOMB_DROPPED_NEXT_TO_CRATE")
                self.logger.debug("BOMB_DROPPED_NEXT_TO_CRATE")
            else:
                events.append(e.BOMB_DROPPED_AWAY_FROM_CRATE)


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
    #   Custom Event: Bomb dropped, check for escape routes and choose a good one
    #

    if is_bomb_under_players_feet(old_game_state):
        self.logger.debug("BOMB_DROPPED ...")
        print("BOMB_DROPPED ...") 
        print(can_escape_left(old_game_state))
        print(can_escape_right(old_game_state))
        print(can_escape_up(old_game_state))
        print(can_escape_down(old_game_state))
        print(self_action)
        if can_escape_left(old_game_state) == 1 and self_action == "LEFT":
            events.append(e.CHOSE_GOOD_ESCAPE)
            self.logger.debug("CHOSE_GOOD_ESCAPE")
            print("CHOSE_GOOD_ESCAPE") 
        elif can_escape_right(old_game_state) == 1 and self_action == "RIGHT":
            events.append(e.CHOSE_GOOD_ESCAPE)
            self.logger.debug("CHOSE_GOOD_ESCAPE")
            print("CHOSE_GOOD_ESCAPE") 
        elif can_escape_down(old_game_state) == 1 and self_action == "DOWN":
            events.append(e.CHOSE_GOOD_ESCAPE)
            self.logger.debug("CHOSE_GOOD_ESCAPE")
            print("CHOSE_GOOD_ESCAPE") 
        elif can_escape_up(old_game_state) == 1 and self_action == "UP":
            events.append(e.CHOSE_GOOD_ESCAPE)
            self.logger.debug("CHOSE_GOOD_ESCAPE")
            print("CHOSE_GOOD_ESCAPE") 
        elif can_escape_left(old_game_state) == 0 and self_action == "LEFT":
            events.append(e.CHOSE_BAD_ESCAPE)
            self.logger.debug("CHOSE_BAD_ESCAPE")
            print("CHOSE_BAD_ESCAPE") 
        elif can_escape_right(old_game_state) == 0 and self_action == "RIGHT":
            events.append(e.CHOSE_BAD_ESCAPE)
            self.logger.debug("CHOSE_BAD_ESCAPE")
            print("CHOSE_BAD_ESCAPE") 
        elif can_escape_down(old_game_state) == 0 and self_action == "DOWN":
            events.append(e.CHOSE_BAD_ESCAPE)
            self.logger.debug("CHOSE_BAD_ESCAPE")
            print("CHOSE_BAD_ESCAPE") 
        elif can_escape_up(old_game_state) == 0 and self_action == "UP":
            events.append(e.CHOSE_BAD_ESCAPE)
            self.logger.debug("CHOSE_BAD_ESCAPE")
            print("CHOSE_BAD_ESCAPE") 
        elif self_action == "WAIT":
            events.append(e.CHOSE_BAD_ESCAPE)
            self.logger.debug("CHOSE_BAD_ESCAPE")
            print("CHOSE_BAD_ESCAPE") 
            

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
        



    
    
    print(f"old_agent_pos: {old_agent_pos}")
    print(f"new_agent_pos: {new_agent_pos}")
    print(f"old_state_index: {old_state_index}")
    print(f"new_state_index: {new_state_index}")
    print(f"self_action: {self_action}")



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
    
    new_value = factor1 + factor3
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
    print(f"LAST ACTION: {last_action}")

    #
    #   Custom event survived bomb cleanup
    #
    active_bombs.clear()


    #
    #
    #   ---   Compute State Index of last_game_state
    #
    #
    
    #   Agent
    last_agent_pos = last_game_state["self"][3]
    last_agent_pos_index = compute_agent_pos_index(last_agent_pos)
    
    #   Coin, Crate, Bomb
    last_closest_coin = find_closest_coin(last_game_state)
    last_closest_crate = find_closest_crate(last_game_state)
    last_closest_bomb = find_closest_bomb(last_game_state)    
    last_coin_pos_index, last_crate_pos_index, last_bomb_pos_index = compute_indices(last_closest_coin, last_closest_crate, last_closest_bomb)
    last_is_bomb_under_players_feet = is_bomb_under_players_feet(last_game_state)

    #   Compute Pull Factor
    last_pull_index = compute_pull_factor_index(last_game_state, last_crate_pos_index, last_coin_pos_index)

    #   Compute Index
    last_state_index = compute_state_index(last_game_state, last_agent_pos_index, last_pull_index, last_bomb_pos_index, last_is_bomb_under_players_feet)


 
    last_action_index = action_to_index[last_action]

    #
    #   Update Q-value of state-action tupel
    #
    reward = reward_from_events(self, events)
    argmax = np.argmax( self.Q[last_state_index] )

    factor1 = ( 1.0 - ALPHA ) * self.Q[last_state_index][last_action_index]
    factor2 = GAMMA * self.Q[last_state_index][argmax]
    factor3 = ALPHA * ( reward + factor2 )
    new_value = factor1 + factor3
    self.Q[last_state_index][last_action_index] = new_value

    print(f"reward: {reward}")

    #
    #   Debugging
    #

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
        #e.COIN_COLLECTED: 2,
        #e.WAITED: -0.2,
        e.INVALID_ACTION: -0.1,
        #e.MOVED_RIGHT: -0.1,
        #e.MOVED_UP: -0.1,
        #e.MOVED_DOWN: -0.1,
        #e.MOVED_LEFT: -0.1,
        #e.BOMB_DROPPED: 0.2,
        #e.CRATE_DESTROYED: .5,
        #e.IN_DANGER: -5,
        e.KILLED_SELF: -3.0,
        e.STEPPED_AWAY_FROM_BOMB: 1.0,
        #e.SURVIVED_BOMB: 1.0,
        e.BOMB_DROPPED_NEXT_TO_CRATE: 1.0,
        e.BOMB_DROPPED_AWAY_FROM_CRATE: -1.0,
        e.CHOSE_GOOD_ESCAPE: 1.0,
        e.CHOSE_BAD_ESCAPE: -1.0
        
        
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            if event == e.SURVIVED_BOMB:
                print("bomb survied lol")
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
