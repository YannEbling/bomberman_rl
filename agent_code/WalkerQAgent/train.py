import pickle
import events as e
from .callbacks import state_to_index
from collections import deque
from typing import List


# Hyperparameters
ALPHA = 0.3
GAMMA = 0.01
TRANSITION_HISTORY_SIZE = 12
EPSILON = 0.05

# Rewards
R_BASE = -0.08
R_COIN = 1


# Event rewards
EVENT_REWARDS = {e.COIN_COLLECTED: R_COIN,
                 e.MOVED_LEFT: R_BASE,
                 e.MOVED_UP: R_BASE,
                 e.MOVED_DOWN: R_BASE,
                 e.MOVED_RIGHT: R_BASE,
                 e.WAITED: R_BASE}

# Action indices
ACTION_INDICES = {e.MOVED_UP: 0,
                  e.MOVED_RIGHT: 1,
                  e.MOVED_DOWN: 2,
                  e.MOVED_LEFT: 3
                  }


# STILL TO WORK OUT:
def setup_training(self):
    """
    I dont know about this history queue, wether it is necessary for Q learning.
    """
    #self.history = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.total_reward = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.current_reward = sum([EVENT_REWARDS[event] for event in events])
    self.logger.info(f"Awarded {self.curent_reward} for events {', '.join(events)}")
    self.total_reward += self.current_reward
    old_game_state_index = state_to_index(old_game_state)
    new_game_state_index = state_to_index(new_game_state)
    #self.history.append((old_game_state_index, self_action, new_game_state_index, self.current_reward, self.total_reward))
    # Learning according to Q learning formula on https://en.wikipedia.org/wiki/Q-learning
    self.model.Q[ACTION_INDICES[self_action], old_game_state_index] *= (1-ALPHA)
    self.model.Q[ACTION_INDICES[self_action], old_game_state_index] += ALPHA * (self.current_reward + GAMMA * max(
        self.model.Q[:, new_game_state_index]
    ))


def end_of_game(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.current_reward = sum([EVENT_REWARDS[event] for event in events])
    self.logger.info(f"Awarded {self.curent_reward} for events {', '.join(events)}")
    self.total_reward += self.current_reward
    last_game_state_index = state_to_index(last_game_state)
    self.model.Q[ACTION_INDICES[last_action]] *= (1-ALPHA)
    self.model.Q[ACTION_INDICES[last_action], last_game_state_index] += ALPHA * self.current_reward

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)