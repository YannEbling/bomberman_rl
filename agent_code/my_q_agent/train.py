from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np
from main import *

from settings import *

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
ALPHA = 0.5


# Discount factor gamma, 0.0 < gamma < 1.0
GAMMA = 0.3


MyTransition = namedtuple('Transition', ('state', 'action'))
MY_TRANSITION_HISTORY_SIZE = 40


# Action to index
action_to_index = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "WAIT": 4
}

# For training debugging
coins_collected = 0
round = 0


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



    #print(test)

    # Idea: Add your own events to hand out rewards
    #if ...:
        #events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    #---
    
    
    #global parser
    #print(parser)

    
    #
    #   Compute State Index of old_game_state
    #
    # (1,1) <= agent_position <= (7,7) = 1 <= index <= 49
    old_agent_pos   = old_game_state["self"][3]
    old_agent_pos_x = old_agent_pos[0]
    old_agent_pos_y = old_agent_pos[1]
    old_agent_pos_index = (old_agent_pos_x - 1 + (COLS - 2) * (old_agent_pos_y - 1)) + 1
    
    # Coin position
    old_coin_pos_index = 1
    
    if len(old_game_state["coins"]) > 0:
        old_coin_pos   = old_game_state["coins"][0]
        old_coin_pos_x = old_coin_pos[0]
        old_coin_pos_y = old_coin_pos[1]
        old_coin_pos_index = (old_coin_pos_x - 1 + (COLS - 2) * (old_coin_pos_y - 1)) + 1
    
    # 0 <= state_index <= 2400
    old_state_index = old_agent_pos_index * (((COLS - 2) * (COLS - 2)) - 1) + old_coin_pos_index - 1
    

    
    #
    #   Compute Action Index of old_game_state
    #
    old_action_index = action_to_index[self_action]
    
    
    
    
    #
    #   Compute State Index of new_game_state
    #
    # (1,1) <= agent_position <= (7,7) = 1 <= index <= 49
    new_agent_pos   = new_game_state["self"][3]
    new_agent_pos_x = new_agent_pos[0]
    new_agent_pos_y = new_agent_pos[1]
    new_agent_pos_index = (new_agent_pos_x - 1 + (COLS - 2)  * (new_agent_pos_y - 1)) + 1
    
    # Coin position
    new_coin_pos_index = 1
    
    if len(new_game_state["coins"]) > 0:
        new_coin_pos   = new_game_state["coins"][0]
        new_coin_pos_x = new_coin_pos[0]
        new_coin_pos_y = new_coin_pos[1]
        new_coin_pos_index = (new_coin_pos_x - 1 + (COLS - 2) * (new_coin_pos_y - 1)) + 1
    
    # 0 <= state_index <= 2400
    new_state_index = new_agent_pos_index * ((COLS - 2) * (COLS - 2) - 1) + new_coin_pos_index - 1
    

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
    
    self.logger.debug(f"np.argmax: {np.argmax(self.Q[new_state_index])}")
    self.logger.debug(f"np.argmax value: {self.Q[new_state_index][argmax]}")
    
    self.logger.debug(f"factor1 :{factor1}")
    self.logger.debug(f"factor2 :{factor2}")
    self.logger.debug(f"factor3 :{factor3}")
    
    self.logger.debug(f"old q value before update: {self.Q[old_state_index][old_action_index]}")
    new_value = factor1 + factor2 + factor3
    self.logger.debug(f"new q value: {new_value}")
    self.logger.debug(f"reward: {reward}")
    
    # set new value
    self.Q[old_state_index][old_action_index] = new_value
    
    self.logger.debug(f"new q value before update: {self.Q[old_state_index][old_action_index]}")
    
    #print(events)
    
    
    
    
    self.mytransitions.append(MyTransition(new_game_state, self_action))
    
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

    #
    #   Compute State Index of new_game_state
    #
    # (1,1) <= agent_position <= (7,7) = 1 <= index <= 49
    last_agent_pos   = last_game_state["self"][3]
    last_agent_pos_x = last_agent_pos[0]
    last_agent_pos_y = last_agent_pos[1]
    last_agent_pos_index = (last_agent_pos_x - 1 + (COLS - 2) * (last_agent_pos_y - 1)) + 1
    
    # Coin position
    last_coin_pos_index = 1
    
    if len(last_game_state["coins"]) > 0:
        last_coin_pos   = last_game_state["coins"][0]
        last_coin_pos_x = last_coin_pos[0]
        last_coin_pos_y = last_coin_pos[1]
        last_coin_pos_index = (last_coin_pos_x - 1 + (COLS - 2) * (last_coin_pos_y - 1)) + 1
    
    # 0 <= state_index <= 2400
    last_state_index = last_agent_pos_index * ((COLS - 2) * (COLS - 2) - 1) + last_coin_pos_index - 1
 
    last_action_index = action_to_index[last_action]

    #
    #   Update Q-value of state-action tupel
    #
    reward = reward_from_events(self, events)
    argmax = np.argmax( self.Q[last_state_index] )

    factor1 = ( 1.0 - ALPHA ) * self.Q[last_state_index][ last_action_index]
    factor2 = GAMMA * self.Q[last_state_index][argmax]
    factor3 = ALPHA * ( reward + factor2 )
    new_value = factor1 + factor2 + factor3

    self.Q[last_state_index][last_action_index] = new_value
    #
    #   Update Q-value of state-action tupel
    #
    # set new value
    #r = reward_from_events(self, events)
    #if r > 1.0:
    #    for i in range(len(self.Q[last_state_index])):
    #        self.Q[last_state_index][i] = r

    #   Debug - coin found?
    for event in events:
        if event == e.COIN_COLLECTED:
            global coins_collected 
            coins_collected = coins_collected + 1
    
    print(f"Coins collected: {coins_collected}")


    for i in range(len(self.Q)):
        self.logger.debug(self.Q[i])

    global round
    round = round + 1
    self.logger.debug(f"Round nr: {round}")




    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Q, file)

    #print(test)

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
        e.COIN_COLLECTED: 10,
        #e.KILLED_OPPONENT: 5,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1
        #PLACEHOLDER_EVENT: -.05  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
