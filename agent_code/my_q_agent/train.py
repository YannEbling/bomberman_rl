from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

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
GAMMA = 0.5


MyTransition = namedtuple('Transition', ('state', 'action'))
MY_TRANSITION_HISTORY_SIZE = 40


# action to index
action_to_index = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "WAIT": 4
}



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
    
    #---
    
    
    
    #
    #   Compute State Index of old_game_state
    #
    # (1,1) <= agent_position <= (7,7) = 1 <= index <= 49
    old_agent_pos   = old_game_state["self"][3]
    old_agent_pos_x = old_agent_pos[0]
    old_agent_pos_y = old_agent_pos[1]
    old_agent_pos_index = (old_agent_pos_x - 1 + 7 * (old_agent_pos_y - 1)) + 1
    
    # Coin position
    old_coin_pos_index = 1
    
    if len(old_game_state["coins"]) > 0:
        old_coin_pos   = old_game_state["coins"][0]
        old_coin_pos_x = old_coin_pos[0]
        old_coin_pos_y = old_coin_pos[1]
        old_coin_pos_index = (old_coin_pos_x - 1 + 7 * (old_coin_pos_y - 1)) + 1
    
    # 0 <= state_index <= 2400
    old_state_index = old_agent_pos_index * old_coin_pos_index - 1
    
    
    
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
    new_agent_pos_index = (new_agent_pos_x - 1 + 7 * (new_agent_pos_y - 1)) + 1
    
    # Coin position
    new_coin_pos_index = 1
    
    if len(new_game_state["coins"]) > 0:
        new_coin_pos   = new_game_state["coins"][0]
        new_coin_pos_x = new_coin_pos[0]
        new_coin_pos_y = new_coin_pos[1]
        new_coin_pos_index = (new_coin_pos_x - 1 + 7 * (new_coin_pos_y - 1)) + 1
    
    # 0 <= state_index <= 2400
    new_state_index = new_agent_pos_index * new_coin_pos_index - 1
    



    #
    #   Update Q-value of state-action tupel
    #
    factor1 = ( 1 - ALPHA ) * self.Q[old_state_index][old_action_index]
    factor2 = GAMMA * np.argmax( self.Q[new_state_index] )
    factor3 = ALPHA * ( reward_from_events(self, events) * factor2 )
    
    # set new value
    self.Q[old_state_index][old_action_index] = factor1 + factor2 + factor3
    
    
    print(events)
    
    
    
    
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
    last_agent_pos_index = (last_agent_pos_x - 1 + 7 * (last_agent_pos_y - 1)) + 1
    
    # Coin position
    last_coin_pos_index = 1
    
    if len(last_game_state["coins"]) > 0:
        last_coin_pos   = last_game_state["coins"][0]
        last_coin_pos_x = last_coin_pos[0]
        last_coin_pos_y = last_coin_pos[1]
        last_coin_pos_index = (last_coin_pos_x - 1 + 7 * (last_coin_pos_y - 1)) + 1
    
    # 0 <= state_index <= 2400
    last_state_index = last_agent_pos_index * last_coin_pos_index - 1
 
    #
    #   Update Q-value of state-action tupel
    #
    # set new value
    r = reward_from_events(self, events)
    if r > 1.0:
        for i in range(len(self.Q[last_state_index])):
            self.Q[last_state_index][i] = reward_from_events(self, events)

    #   Debug - coin found?










    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Q, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        #e.KILLED_OPPONENT: 5,
        e.WAITED: -1,
        e.INVALID_ACTION: -0.5,
        e.MOVED_RIGHT: -0.05,
        e.MOVED_UP: -0.05,
        e.MOVED_DOWN: -0.05,
        e.MOVED_LEFT: -0.05
        #PLACEHOLDER_EVENT: -.05  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
