from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import find_closest_coin
from .callbacks import find_closest_crate
from .callbacks import custom_bomb_state
from .callbacks import find_closest_bomb
from .callbacks import is_bomb_under_players_feet
from .callbacks import can_escape_down
from .callbacks import can_escape_up
from .callbacks import can_escape_left
from .callbacks import can_escape_right
from .callbacks import compute_evade_bomb_index
from .callbacks import update_index_including_bomb_evade_index

import math

import numpy as np
from main import *

import time

from settings import *

import os
from . import auxiliary_functions as aux


# -----------------------------------------------------------
#      !!! IMPORTANT !!!
# -----------------------------------------------------------
# Saving data every round is costly, save only every n rounds
# SAVE_INTERVAL must match the value for the parameter --n-rounds
# (Would read this out of args but it somehow doesnt work) 
#




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
ALPHA = 0.2


# Discount factor gamma, 0.0 < gamma < 1.0
GAMMA = 0.9


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
round = 0

id = os.getpid()

# convert setting values to used values in training
cols = (COLS - 2)
cells = pow((COLS - 2), 2)

# Execution Time Analysis

# Saving data every round is costly, save only every n rounds
SAVE_INTERVAL = 1000
save_counter = 0

# This function and the VALID_POSITIONS are necessary to perform the sped up training, looping over all possible
# bomb_positions
def condition(i: int, j: int, n: int) -> bool:
    con1 = i % 2 != 0 or j % 2 != 0
    con2 = i != 0 and j != 0
    con3 = i != n and j != n
    return con1 and con2 and con3


N = cols+1
DOMAIN = np.arange(0, 17)
VALID_POSITIONS = [(i, j) for i in DOMAIN for j in DOMAIN if condition(i, j, N)] + [(0, 0)]

BOMB_EVADE_STATES = 16

round_in_game = 0

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
    





    #print(id)




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
    
    global round_in_game
    round_in_game = round_in_game + 1
    self.logger.debug(f"TR: round: {round_in_game}")

    

    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action,
    #                                   state_to_features(new_game_state), reward_from_events(self, events)))
    
    #---
    
    #
    #   Compute State Index of old_game_state
    #
    old_agent_pos = old_game_state["self"][3]

    


    # Coin and bomb position
    old_possible_closest_coin_index = aux.index_of_closest_item(old_agent_pos, old_game_state['coins'])
    old_possible_closest_bomb_index = aux.index_of_closest_item(old_agent_pos, [custom_bomb_state[k][0] for k in
                                                                                range(len(custom_bomb_state))])

    old_closest_enemy_index = aux.index_of_closest_item(old_agent_pos, [old_game_state['others'][k][3] for k in range(len(old_game_state['others']))])
    
    old_use_enemy = False
    old_use_crate = False
    old_closest_crate = None
    old_enemy_pos = None

    if old_closest_enemy_index != None:
        old_enemy_pos = old_game_state['others'][old_closest_enemy_index][3]
        if old_possible_closest_coin_index != None:
            coin_pos = old_game_state['coins'][old_possible_closest_coin_index]
            dist_to_coin = math.sqrt(pow((coin_pos[0] - old_agent_pos[0]), 2) + pow((coin_pos[1] - old_agent_pos[1]), 2))
            dist_to_enemy = math.sqrt(pow((old_enemy_pos[0] - old_agent_pos[0]), 2) + pow((old_enemy_pos[1] - old_agent_pos[1]), 2))
            if dist_to_enemy < dist_to_coin:
                old_use_enemy = True
                old_possible_closest_coin_index = old_closest_enemy_index
            
            #print("")
            #print("coin")
            #print(coin_pos)
            #print(dist_to_coin)
            #print("enemy")
            #print(old_enemy_pos)
            #print(dist_to_enemy)
            
        else:
            old_possible_closest_coin_index = old_closest_enemy_index
            old_use_enemy = True
            
        if e.KILLED_OPPONENT in events:
            #print("opponent killed!")
            pass
    elif old_possible_closest_coin_index == None:
        old_closest_crate = find_closest_crate(old_game_state)
        old_use_crate = True
        
    # 0 <= state_index < 790.128
    old_state_index, permutations = aux.state_to_index(game_state=old_game_state,
                                                       custom_bomb_state=custom_bomb_state,
                                                       coin_index=old_possible_closest_coin_index,
                                                       bomb_index=old_possible_closest_bomb_index,
                                                       dim_reduce=True,
                                                       include_bombs=True,
                                                       include_crates=True,
                                                       enemy_instead_of_coin=old_use_enemy, 
                                                       crate=old_closest_crate, 
                                                       crate_instead_of_coin_or_enemy=old_use_crate)
    
    old_state_index = update_index_including_bomb_evade_index(old_state_index, old_game_state)
    
    permuted_old_action = aux.apply_permutations(self_action, permutations)
    
    #
    #   Compute Action Index of old_game_state
    #
    old_action_index = action_to_index[permuted_old_action]

    #
    #   Evade bomb, if on bomb
    #

    #
    #   Custom Event: Bomb dropped, check for escape routes and choose a good one
    #



    #print(f"old_state_index:                         {old_state_index}")
    #print(f"update_index_including_bomb_evade_index: {update_index_including_bomb_evade_index(old_state_index, old_game_state)}")




    self.logger.debug(f"TR: old training")
    self.logger.debug(f"TR: Q[{old_state_index}]: {self.Q[old_state_index]}")
    self.logger.debug(f"TR: self_action: {self_action}")
    self.logger.debug(f"TR: Action chosen after reverting permutations {permutations}: {permuted_old_action}")



    #
    #   Compute State Index of new_game_state
    #

    new_agent_pos = new_game_state["self"][3]
    


    # Coin and bomb position
    new_possible_closest_coin_index = aux.index_of_closest_item(new_agent_pos, new_game_state['coins'])
    new_possible_closest_bomb_index = aux.index_of_closest_item(new_agent_pos, [custom_bomb_state[k][0] for k in
                                                                                range(len(custom_bomb_state))])
    
    new_closest_enemy_index = aux.index_of_closest_item(new_agent_pos, [new_game_state['others'][k][3] for k in range(len(new_game_state['others']))])
        
    
    new_use_enemy = False
    new_use_crate = False
    new_closest_crate = None
    new_enemy_pos = None
    
    if new_closest_enemy_index != None:
        new_enemy_pos = new_game_state['others'][new_closest_enemy_index][3]
        if new_possible_closest_coin_index != None:
            coin_pos = new_game_state['coins'][new_possible_closest_coin_index]
            dist_to_coin = math.sqrt(pow((coin_pos[0] - new_agent_pos[0]), 2) + pow((coin_pos[1] - new_agent_pos[1]), 2))
            dist_to_enemy = math.sqrt(pow((new_enemy_pos[0] - new_agent_pos[0]), 2) + pow((new_enemy_pos[1] - new_agent_pos[1]), 2))
            if dist_to_enemy < dist_to_coin:
                new_use_enemy = True
                new_possible_closest_coin_index = new_closest_enemy_index
            
            #print("")
            #print("coin")
            #print(coin_pos)
            #print(dist_to_coin)
            #print("enemy")
            #print(new_enemy_pos)
            #print(dist_to_enemy)
            
        else:
            new_possible_closest_coin_index = new_closest_enemy_index
            new_use_enemy = True
    elif new_possible_closest_coin_index == None:
        new_closest_crate = find_closest_crate(new_game_state)
        new_use_crate = True
    
    # 0 <= state_index <= 790.128
    new_state_index = aux.state_to_index(game_state=new_game_state,
                                         custom_bomb_state=custom_bomb_state,
                                         coin_index=new_possible_closest_coin_index,
                                         bomb_index=new_possible_closest_bomb_index,
                                         dim_reduce=True,
                                         include_bombs=True,
                                         include_crates=True,
                                         enemy_instead_of_coin=new_use_enemy, 
                                         crate=new_closest_crate, 
                                         crate_instead_of_coin_or_enemy=new_use_crate
                                         )[0]

    new_state_index = update_index_including_bomb_evade_index(new_state_index, new_game_state)
  

#    if is_bomb_under_players_feet(new_game_state):
#        new_state_index = len(self.Q) - BOMB_EVADE_STATES + compute_evade_bomb_index(new_game_state)

    # Add custom event: walking into or out of bomb explosion radius (in or out danger)
    old_in_danger = in_danger(agent_position=old_agent_pos,
                              bombs=custom_bomb_state,
                              bomb_index=old_possible_closest_bomb_index)
    new_in_danger = in_danger(agent_position=new_agent_pos,
                              bombs=custom_bomb_state,
                              bomb_index=new_possible_closest_bomb_index)

    if old_in_danger and not new_in_danger:
        events.append("OUT_DANGER")

    if not old_in_danger and new_in_danger:
        if self_action != 'BOMB':
            events.append("IN_DANGER")


    if is_bomb_under_players_feet(old_game_state):
        self.logger.debug("BOMB_DROPPED ...")
        #print(f"bomb dropped: self_action chosen {self_action}")
        if can_escape_left(old_game_state) == 1 and old_agent_pos[0] == new_agent_pos[0] + 1:
            events.append(e.CHOSE_GOOD_ESCAPE)
            #self.logger.debug("CHOSE_GOOD_ESCAPE")
            #print("CHOSE_GOOD_ESCAPE")
        elif can_escape_right(old_game_state) == 1 and old_agent_pos[0] == new_agent_pos[0] - 1:
            events.append(e.CHOSE_GOOD_ESCAPE)
            #self.logger.debug("CHOSE_GOOD_ESCAPE")
            #print("CHOSE_GOOD_ESCAPE")
        elif can_escape_down(old_game_state) == 1 and 0 and old_agent_pos[1] == new_agent_pos[1] - 1:
            events.append(e.CHOSE_GOOD_ESCAPE)
            #self.logger.debug("CHOSE_GOOD_ESCAPE")
            #print("CHOSE_GOOD_ESCAPE")
        elif can_escape_up(old_game_state) == 1 and 0 and old_agent_pos[1] == new_agent_pos[1] + 1:
            events.append(e.CHOSE_GOOD_ESCAPE)
            #self.logger.debug("CHOSE_GOOD_ESCAPE")
            #print("CHOSE_GOOD_ESCAPE")
        elif can_escape_left(old_game_state) == 0 and old_agent_pos[0] == new_agent_pos[0] + 1:
            events.append(e.CHOSE_BAD_ESCAPE)
            #self.logger.debug("CHOSE_BAD_ESCAPE")
            #print("CHOSE_BAD_ESCAPE")
        elif can_escape_right(old_game_state) == 0 and old_agent_pos[0] == new_agent_pos[0] - 1:
            events.append(e.CHOSE_BAD_ESCAPE)
            #self.logger.debug("CHOSE_BAD_ESCAPE")
            #print("CHOSE_BAD_ESCAPE")
        elif can_escape_down(old_game_state) == 0 and old_agent_pos[1] == new_agent_pos[1] - 1:
            events.append(e.CHOSE_BAD_ESCAPE)
            #self.logger.debug("CHOSE_BAD_ESCAPE")
            #print("CHOSE_BAD_ESCAPE")
        elif can_escape_up(old_game_state) == 0 and old_agent_pos[1] == new_agent_pos[1] + 1:
            events.append(e.CHOSE_BAD_ESCAPE)
            #self.logger.debug("CHOSE_BAD_ESCAPE")
            #print("CHOSE_BAD_ESCAPE")


    #
    #   Custom Event: Bomb placed directly at crate?
    #

    old_possible_closest_crate = find_closest_crate(old_game_state)
    self.logger.debug(f"TR: old crate pos: {old_possible_closest_crate}")
    self.logger.debug(f"TR: old agent pos: {old_agent_pos}")
    if old_possible_closest_crate is not None:
        for event in events:
            if event == e.BOMB_DROPPED:
                dist_to_crate = math.sqrt(pow((old_possible_closest_crate[0] - old_agent_pos[0]), 2) + pow((old_possible_closest_crate[1] - old_agent_pos[1]), 2))
                if dist_to_crate <= 1.01:
                    events.append(e.BOMB_DROPPED_NEXT_TO_CRATE)
                    self.logger.debug("TR: BOMB_DROPPED_NEXT_TO_CRATE")
                else:
                    events.append(e.BOMB_DROPPED_AWAY_FROM_CRATE)
                    self.logger.debug("TR: BOMB_DROPPED_AWAY_FROM_CRATE")



    if old_enemy_pos != None:
        dist_to_enemy = math.sqrt(pow((old_enemy_pos[0] - new_agent_pos[0]), 2) + pow((old_enemy_pos[1] - new_agent_pos[1]), 2))
        if dist_to_enemy < 1.01 and old_use_enemy:
            print("touched enemy!")
            events.append(e.TOUCHED_ENEMY)

    with open("output.txt", "a") as file:
        output = f"old_q: {self.Q[old_state_index]}"
        file.write(output)
        
        
    if e.BOMB_EXPLODED in events:
        events.append(e.SURVIVED_BOMB_EXPLOSION)
    
    if e.BOMB_DROPPED in events:
        if old_enemy_pos != None:
            bomb_dropped_dist_to_enemy = math.sqrt(pow((old_enemy_pos[0] - old_agent_pos[0]), 2) + pow((old_enemy_pos[1] - old_agent_pos[1]), 2))
            if bomb_dropped_dist_to_enemy <= 1:
                events.append(e.BOMB_DROPPED_NEXT_TO_ENEMY)
                #print("bomb dropped next to enemy agent!")
    


    #
    #   Custom Event:
    #
    if old_possible_closest_bomb_index == None or math.sqrt(pow((custom_bomb_state[old_possible_closest_bomb_index][0][0] - old_agent_pos[0]), 2) + pow((custom_bomb_state[old_possible_closest_bomb_index][0][1] - old_agent_pos[1]), 2)) > 4.01:
        if old_use_crate:
            if old_closest_crate != None:
                old_di_ag_cr = math.sqrt(pow((old_possible_closest_crate[0] - old_agent_pos[0]), 2) + pow((old_possible_closest_crate[1] - old_agent_pos[1]), 2))
                new_di_ag_cr = math.sqrt(pow((old_possible_closest_crate[0] - new_agent_pos[0]), 2) + pow((old_possible_closest_crate[1] - new_agent_pos[1]), 2))
                if old_di_ag_cr > new_di_ag_cr:
                    events.append(e.STEPPED_TOWARDS_POINT_OF_INTEREST)
                    #print("STEPPED_TOWARDS_POINT_OF_INTEREST - crate")
        elif old_use_enemy:
            if old_enemy_pos != None:
                old_di_ag_en = math.sqrt(pow((old_enemy_pos[0] - old_agent_pos[0]), 2) + pow((old_enemy_pos[1] - old_agent_pos[1]), 2))
                new_di_ag_en = math.sqrt(pow((old_enemy_pos[0] - new_agent_pos[0]), 2) + pow((old_enemy_pos[1] - new_agent_pos[1]), 2))
                if old_di_ag_en > new_di_ag_en:
                    events.append(e.STEPPED_TOWARDS_POINT_OF_INTEREST)
                    #print("STEPPED_TOWARDS_POINT_OF_INTEREST - enemy")
        else:
            if old_possible_closest_coin_index != None:
                old_coin_pos = old_game_state['coins'][old_possible_closest_coin_index]
                old_di_ag_co = math.sqrt(pow((old_coin_pos[0] - old_agent_pos[0]), 2) + pow((old_coin_pos[1] - old_agent_pos[1]), 2))
                new_di_ag_co = math.sqrt(pow((old_coin_pos[0] - new_agent_pos[0]), 2) + pow((old_coin_pos[1] - new_agent_pos[1]), 2))
                if old_di_ag_co > new_di_ag_co:
                    events.append(e.STEPPED_TOWARDS_POINT_OF_INTEREST)
                    #print("STEPPED_TOWARDS_POINT_OF_INTEREST - coin")


    #
    #   Custom event: Stepped away from bomb
    #
    if old_possible_closest_bomb_index != None:
        old_closest_bomb = custom_bomb_state[old_possible_closest_bomb_index]
        if old_closest_bomb != None:
            #print(f"old_agent_pos: {old_agent_pos}")
            #print(f"new_agent_pos: {new_agent_pos}")

            old_dist_to_bomb = math.sqrt(pow((old_closest_bomb[0][0] - old_agent_pos[0]), 2) + pow((old_closest_bomb[0][1] - old_agent_pos[1]), 2))
            new_dist_to_bomb = math.sqrt(pow((old_closest_bomb[0][0] - new_agent_pos[0]), 2) + pow((old_closest_bomb[0][1] - new_agent_pos[1]), 2))

            #print(f"old_dist_to_bomb: {old_dist_to_bomb}")
            #print(f"new_dist_to_bomb: {new_dist_to_bomb}")

            if new_dist_to_bomb > old_dist_to_bomb:
                #print("stepped away from bomb")
                events.append(e.STEPPED_AWAY_FROM_BOMB)

            elif new_dist_to_bomb < old_dist_to_bomb:
                #print("stepped towards bomb")
                events.append(e.STEPPED_TOWARDS_BOMB)



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
    new_value = factor1 + factor3


    





    #self.logger.debug(f"np.argmax: {np.argmax(self.Q[new_state_index])}")
    #self.logger.debug(f"np.argmax value: {self.Q[new_state_index][argmax]}")

    #self.logger.debug(f"factor1 :{factor1}")
    #self.logger.debug(f"factor2 :{factor2}")
    #self.logger.debug(f"factor3 :{factor3}")

    #self.logger.debug(f"old q value before update: {self.Q[old_state_index][old_action_index]}")
    #self.logger.debug(f"new q value: {new_value}")
    self.logger.debug(f"TR: reward: {reward}")
    
    # set new value
    self.Q[old_state_index][old_action_index] = new_value



    if round_in_game == 1:
        #print("new round")
        with open("output.txt", "w") as file:
            file.write("")

    with open("output.txt", "a") as file:
    # Write the string to the file
        #if old_action_index == 5:
        #if permuted_old_action == "BOMB":
        output = f"""
        permutated action: {permuted_old_action}
        self action: {self_action}
        state index: {old_state_index}
        reward: {reward}
        events: {events}
        updated_old_q: {self.Q[old_state_index]}
        
        """
        file.write(output)




    self.logger.debug(f"TR: new q after update: {self.Q[old_state_index]}")

 # to speed up training, a virtual bomb is placed for every valid position. Then, the rewards that the chosen action
    # would have resulted in are recorded and the Q matrix is updated.
    #for bomb_position in VALID_POSITIONS:
    #    old_virtual_state = {
    #                        'field': old_game_state['field'],
    #                        'self': old_game_state['self'],
    #                        'others': old_game_state['others'],
    #                        'bombs': [[bomb_position]],
    #                        'coins': old_game_state['coins']
    #                        }

    #    new_virtual_state = {
    #                        'field': new_game_state['field'],
    #                        'self': new_game_state['self'],
    #                        'others': new_game_state['others'],
    #                        'bombs': [[bomb_position]],
    #                        'coins': new_game_state['coins']
    #                        }

    #    old_state_index = aux.state_to_index(old_virtual_state,
    #                                         custom_bomb_state=custom_bomb_state,
    #                                         coin_index=old_possible_closest_coin_index,
    #                                         bomb_index=0,
    #                                         dim_reduce=True,
    #                                         include_bombs=True,
    #                                         include_crates=True)[0]

    #    new_state_index = aux.state_to_index(game_state=new_virtual_state,
    #                                         custom_bomb_state=custom_bomb_state,
    #                                         coin_index=new_possible_closest_coin_index,
    #                                         bomb_index=0,
    #                                         dim_reduce=True,
    #                                         include_bombs=True,
    #                                         include_crates=True)[0]

        #
        #   Update Q-value of state-action tupel
        #
    #    reward = reward_from_events(self, events)
    #    argmax = np.argmax(self.Q[new_state_index])#

    #    factor1 = (1.0 - ALPHA) * self.Q[old_state_index][old_action_index]
    #    factor2 = GAMMA * self.Q[new_state_index][argmax]
    #    factor3 = ALPHA * (reward + factor2)

    #    new_value = factor1 + factor3

    #    self.Q[old_state_index][old_action_index] = new_value


   
    
    #---


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

    #training_time = time.time()

    global round_in_game
    round_in_game = round_in_game  + 1
    self.logger.debug(f"TR: round: {round_in_game}")

    #
    #   Compute State Index of new_game_state
    #
    # (1,1) <= agent_position <= (7,7) = 1 <= index <= 49
    last_agent_pos   = last_game_state["self"][3]

    # Coin and bomb position
    possible_closest_coin_index = aux.index_of_closest_item(last_agent_pos, last_game_state['coins'])
    possible_closest_bomb_index = aux.index_of_closest_item(last_agent_pos, [custom_bomb_state[k][0] for k in
                                                                             range(len(custom_bomb_state))])


    last_closest_enemy_index = aux.index_of_closest_item(last_agent_pos, [last_game_state['others'][k][3] for k in range(len(last_game_state['others']))])

    last_use_enemy = False
    last_use_crate = False
    last_closest_crate = None
    last_enemy_pos = None
    
    if last_closest_enemy_index != None:
        last_enemy_pos = last_game_state['others'][last_closest_enemy_index][3]
        if possible_closest_coin_index != None:
            coin_pos = last_game_state['coins'][possible_closest_coin_index]
            dist_to_coin = math.sqrt(pow((coin_pos[0] - last_agent_pos[0]), 2) + pow((coin_pos[1] - last_agent_pos[1]), 2))
            dist_to_enemy = math.sqrt(pow((last_enemy_pos[0] - last_agent_pos[0]), 2) + pow((last_enemy_pos[1] - last_agent_pos[1]), 2))
            if dist_to_enemy < dist_to_coin:
                last_use_enemy = True
                possible_closest_coin_index = last_closest_enemy_index
            
            #print("")
            #print("coin")
            #print(coin_pos)
            #print(dist_to_coin)
            #print("enemy")
            #print(last_enemy_pos)
            #print(dist_to_enemy)
            
        else:
            possible_closest_coin_index = last_closest_enemy_index
            last_use_enemy = True
    elif possible_closest_coin_index == None:
        last_closest_crate = find_closest_crate(last_game_state)
        last_use_crate = True
    
    # 0 <= state_index < 790.128
    last_state_index, permutations = aux.state_to_index(game_state=last_game_state,
                                                        custom_bomb_state=custom_bomb_state,
                                                        coin_index=possible_closest_coin_index,
                                                        bomb_index=possible_closest_bomb_index,
                                                        dim_reduce=True,
                                                        include_bombs=True,
                                                        include_crates=True,
                                                        enemy_instead_of_coin=last_use_enemy,
                                                        crate=last_closest_crate, 
                                                        crate_instead_of_coin_or_enemy=last_use_crate)

    last_state_index = update_index_including_bomb_evade_index(last_state_index, last_game_state)

    permuted_last_action = aux.apply_permutations(last_action, permutations)
    last_action_index = action_to_index[permuted_last_action]



    with open("output.txt", "a") as file:
        output = f"""
        permutated action: {permuted_last_action}
        self action: {last_action}
        state index: {last_state_index}
        events: {events}
        old_last_q: {self.Q[last_state_index]}
        """
        file.write(output)












    #
    #   Update Q-value of state-action tupel
    #
    reward = reward_from_events(self, events)
    argmax = np.argmax( self.Q[last_state_index] )

    factor1 = ( 1.0 - ALPHA ) * self.Q[last_state_index][ last_action_index]
    factor2 = GAMMA * self.Q[last_state_index][argmax]
    factor3 = ALPHA * ( reward + factor2 )
    new_value = factor1 + factor3



    #
    #   Update Q
    #
    self.Q[last_state_index][last_action_index] = new_value

    self.logger.debug(f'TR: Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    #
    # Store the model
    #
    global save_counter
    save_counter = save_counter + 1
    #print(f"save counter: {save_counter}")
    if save_counter == SAVE_INTERVAL:
        #print("saving data")
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
    #exec_time = time.time() - training_time
    #print(f"Execution time for function \"game_events_occurred\": {exec_time:.6f} seconds")
    #self.logger.debug(f"Execution time for function \"game_events_occurred\": {exec_time} seconds")

    with open("output.txt", "a") as file:
    # Write the string to the file
        #if old_action_index == 5:
        #if permuted_old_action == "BOMB":
        output = f"""
        permutated action: {permuted_last_action}
        self action: {last_action}
        state index: {last_state_index}
        reward: {reward}
        events: {events}
        last_q: {self.Q[last_state_index]}
        """
        file.write(output)

 
    round_in_game = 0
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
        e.COIN_COLLECTED: 100,
        #e.KILLED_OPPONENT: 5,
        e.WAITED: -1.0,
        e.INVALID_ACTION: -10.0,
        e.MOVED_RIGHT: -0.2,
        e.MOVED_UP: -0.2,
        e.MOVED_DOWN: -0.2,
        e.MOVED_LEFT: -0.2,
        #PLACEHOLDER_EVENT: -.05  # idea: the custom event is bad
        "IN_DANGER": -10,
        "OUT_DANGER": 10,
        e.KILLED_SELF: -200,
        e.GOT_KILLED: -200,
        e.BOMB_DROPPED_NEXT_TO_CRATE: 50,  # no guarantee, that the reward is sufficient
        e.BOMB_DROPPED_AWAY_FROM_CRATE: -60,
        e.TOUCHED_ENEMY: 50,
        e.KILLED_OPPONENT: 800,
        e.CHOSE_GOOD_ESCAPE: 20.0,
        e.CHOSE_BAD_ESCAPE: -25.0,
        e.BOMB_DROPPED_NEXT_TO_ENEMY: 100,
        #e.SURVIVED_BOMB_EXPLOSION: 80,
        e.STEPPED_TOWARDS_BOMB: -12,
        e.STEPPED_AWAY_FROM_BOMB: 10,
        e.STEPPED_TOWARDS_POINT_OF_INTEREST: 5,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"TR: Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def in_danger(agent_position: tuple, bombs: list[tuple], bomb_index: any) -> bool:
    if bomb_index is None:
        return False

    agent_x, agent_y = agent_position
    bomb_x, bomb_y = bombs[bomb_index][0]
    if bomb_x % 2 != 0 and bomb_y % 2 != 0:
        if agent_x == bomb_x and abs(agent_y - bomb_y) < 4:
            return True
        elif agent_y == bomb_y and abs(agent_x - bomb_x) < 4:
            return True
    elif bomb_x % 2 == 0 and bomb_y % 2 != 0:
        if agent_y == bomb_y and abs(agent_x - bomb_x) < 4:
            return True
    elif bomb_x % 2 != 0 and bomb_y % 2 == 0:
        if agent_x == bomb_x and abs(agent_y - bomb_y) < 4:
            return True
    return False



