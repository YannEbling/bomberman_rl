import pickle
import events as e
from .callbacks import state_to_index
from collections import deque
from typing import List
import csv


SAVING_INTERVALL = 10000

# Hyperparameters
ALPHA = 0.4
GAMMA = 0.2
TRANSITION_HISTORY_SIZE = 12
EPSILON = 0.05
N_EVALSTEPS = 10

# Rewards
R_BASE = -0.05
R_IDLE = -0.3
R_COIN = 10
R_INVALID = -0.4

# custom Reward multiplier
R_DIST = 0.2


# Event rewards
EVENT_REWARDS = {e.COIN_COLLECTED: R_COIN,
                 e.MOVED_LEFT: R_BASE,
                 e.MOVED_UP: R_BASE,
                 e.MOVED_DOWN: R_BASE,
                 e.MOVED_RIGHT: R_BASE,
                 e.WAITED: R_IDLE,
                 e.INVALID_ACTION: R_INVALID,
                 e.SURVIVED_ROUND: 0
                 }

# Action indices
ACTION_INDICES = {'WAIT': 0,
                  'UP': 1,
                  'RIGHT': 2,
                  'DOWN': 3,
                  'LEFT': 4
                  }


# STILL TO WORK OUT:
def setup_training(self):
    """
    I dont know about this history queue, wether it is necessary for Q learning.
    """
    #self.history = deque(maxlen=TRANSITION_HISTORY_SIZE)
    file = open("training_statistic.csv", "w")
    file.close()
    self.total_reward = 0
    self.saving_counter = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.current_reward = 0
    distance_reward = 0
    if old_game_state["step"] == 2:
        self.old_distance = distance_from_coin(old_game_state)
    # custom game event occurs every N_EVALSTEPS steps. If this event is active, the current distance from the coin is
    # compared to the one from the last custom event. Getting closer is rewarded, straying away is penalized.
    if old_game_state["step"] > 2:
        self.new_distance = distance_from_coin(old_game_state)
        distance_reward = R_DIST * (self.old_distance - self.new_distance)
        if distance_reward < 0:
            distance_reward *= 1.2
        self.current_reward += distance_reward
        self.logger.info(f"Awarded {self.current_reward} for distance to coin.")
        self.old_distance = self.new_distance
    for event in events:
        self.current_reward += EVENT_REWARDS[event]
    self.logger.info(f"Awarded {self.current_reward-distance_reward} for events {', '.join(events)}")
    self.total_reward += self.current_reward
    old_game_state_index = state_to_index(old_game_state)
    new_game_state_index = state_to_index(new_game_state)
    # self.history.append((old_game_state_index, self_action, new_game_state_index, self.current_reward, self.total_reward))
    # Learning according to Q learning formula on https://en.wikipedia.org/wiki/Q-learning
    self.model.Q[ACTION_INDICES[self_action], old_game_state_index] *= (1-ALPHA)
    self.model.Q[ACTION_INDICES[self_action], old_game_state_index] += ALPHA * (self.current_reward + GAMMA * max(
        self.model.Q[:, new_game_state_index]
    ))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.current_reward = 0
    # custom game event occurs every N_EVALSTEPS steps. If this event is active, the current distance from the coin is
    # compared to the one from the last custom event. Getting closer is rewarded, straying away is penalized.
    if last_game_state["step"] % N_EVALSTEPS == 0:
        self.new_distance = distance_from_coin(last_game_state)
        self.current_reward += R_DIST * (self.old_distance - self.new_distance)
        self.logger.info(f"Awarded {self.current_reward} for distance to coin")
    for event in events:
        self.current_reward += EVENT_REWARDS[event]
    self.logger.info(f"Awarded {self.current_reward} for events {', '.join(events)}")
    self.total_reward += self.current_reward
    last_game_state_index = state_to_index(last_game_state)
    self.model.Q[ACTION_INDICES[last_action], last_game_state_index] *= (1-ALPHA)
    self.model.Q[ACTION_INDICES[last_action], last_game_state_index] += ALPHA * self.current_reward

    with open("training_statistic.csv", "a") as statistic_file:
        stat_writer = csv.writer(statistic_file)
        stat_writer.writerow([last_game_state['round'], last_game_state['step'], self.total_reward])

    self.total_reward = 0

    self.saving_counter += 1
    if self.saving_counter % SAVING_INTERVALL == 0:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)




def distance_from_coin(game_state: dict):
    coin_x, coin_y = game_state["coins"][-1]
    self_x, self_y = game_state["self"][-1]
    return abs(coin_x - self_x) + abs(coin_y - self_y)
