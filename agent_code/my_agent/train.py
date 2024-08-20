from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

MyTransition = namedtuple('Transition', ('state', 'action'))
MY_TRANSITION_HISTORY_SIZE = 400

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
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    #---
    
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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    
    #---
    # for scenario one-coin only
    # if a coin was collected, modify all action chances for every action executed this round
    
    if self.mytransitions[-1].state['self'][1] == 1:
        print("huay")
        for j, i in enumerate(self.mytransitions):
            agent_pos = i.state['self'][3] 
            agent_pos_x = agent_pos[0]
            agent_pos_y = agent_pos[1]
            agent_pos_index = (agent_pos_x - 1 + 7 * (agent_pos_y - 1)) + 1
    
            coins = i.state['coins']
            
            if len(coins) == 0:
                continue
                
            coin_pos = coins[0]
            coin_pos_x = coin_pos[0]
            coin_pos_y = coin_pos[1]
            coin_pos_index = (coin_pos_x - 1 + 7 * (coin_pos_y - 1)) + 1
  
            index = agent_pos_index * coin_pos_index - 1
            action_chances = self.actions_chances[index]
            
            
            
            match i.action:
                case 'UP':
                    self.actions_chances[index][0] *= 1.5 
                    self.actions_chances[index][1] *= 0.8333 
                    self.actions_chances[index][2] *= 0.8333
                    self.actions_chances[index][3] *= 0.8333
                    self.actions_chances[index][4] *= 0.8333
                
                case 'RIGHT':
                    self.actions_chances[index][0] *= 0.8333
                    self.actions_chances[index][1] *= 1.5 
                    self.actions_chances[index][2] *= 0.8333
                    self.actions_chances[index][3] *= 0.8333
                    self.actions_chances[index][4] *= 0.8333
                
                case 'DOWN':
                    self.actions_chances[index][0] *= 0.8333
                    self.actions_chances[index][1] *= 0.8333
                    self.actions_chances[index][2] *= 1.5 
                    self.actions_chances[index][3] *= 0.8333
                    self.actions_chances[index][4] *= 0.8333
                
                case 'LEFT':
                    self.actions_chances[index][0] *= 0.8333
                    self.actions_chances[index][1] *= 0.8333
                    self.actions_chances[index][2] *= 0.8333
                    self.actions_chances[index][3] *= 1.5 
                    self.actions_chances[index][4] *= 0.8333
                
                case 'WAIT':
                    self.actions_chances[index][0] *= 0.8333
                    self.actions_chances[index][1] *= 0.8333
                    self.actions_chances[index][2] *= 0.8333
                    self.actions_chances[index][3] *= 0.8333
                    self.actions_chances[index][4] *= 1.5 
                    
                
                case _:
                    print("this match case should not be accesible")
            
            
        
    else:
        print("oooh...")
    
    
    #---

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.actions_chances, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
