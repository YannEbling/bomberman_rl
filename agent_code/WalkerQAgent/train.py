import pickle
import events as e
from typing import List
import csv

import auxiliary_functions as aux
import settings as s

SAVING_INTERVAL = 10000

# Hyperparameters
ALPHA = 0.4
GAMMA = 0.2
EPSILON = 0.15
N_EVALSTEPS = 10
NEG_DIST_MULTIPLIER = 1

# Rewards
R_BASE = 0
R_IDLE = 0
R_COIN = 1
R_INVALID = 0

# custom Reward multiplier
R_DIST = 0.1


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


ACTIONS = ['WAIT', 'UP', 'RIGHT', 'DOWN', 'LEFT']


# Action indices
ACTION_INDICES = {'WAIT': 0,
                  'UP': 1,
                  'RIGHT': 2,
                  'DOWN': 3,
                  'LEFT': 4
                  }


PERMUTATIONS = {"none": [0, 1, 2, 3, 4],
                "vertical": [0, 1, 4, 3, 2],
                "horizontal": [0, 3, 2, 1, 4],
                "diagonal": [0, 4, 3, 2, 1]
                }


# STILL TO WORK OUT:
def setup_training(self):
    """
    The training is set up, the statistics file is created empty, the rewards are initialized to 0.
    """
    file = open("training_statistic.csv", "w")
    file.close()
    self.total_reward = 0
    self.saving_counter = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    In each training step, this function is called and evaluates the gained reward from the game states and chosen
    actions. It updates the models Q matrix, corresponding to the formula of Q learning.
    """

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.current_reward = 0
    distance_reward = 0

    coin_index = aux.index_of_closest_coin(old_game_state['self'][-1], old_game_state['coins'])

    if old_game_state["step"] == 2:
        self.old_distance = aux.distance_from_coin(old_game_state["self"][-1], old_game_state["coins"][coin_index])
    # custom game event occurs every step. If this event is active, the current distance from the coin is
    # compared to the one from the last custom event. Getting closer is rewarded, straying away is penalized.
    if old_game_state["step"] > 2:
        self.new_distance = aux.distance_from_coin(old_game_state["self"][-1], old_game_state["coins"][coin_index])
        distance_reward = R_DIST * (self.old_distance - self.new_distance)
        if distance_reward < 0:
            distance_reward *= NEG_DIST_MULTIPLIER
        self.current_reward += distance_reward
        self.logger.info(f"Awarded {self.current_reward} for distance to coin.")
        self.old_distance = self.new_distance
    for event in events:
        self.current_reward += EVENT_REWARDS[event]
    self.logger.info(f"Awarded {self.current_reward-distance_reward} for events {', '.join(events)}")
    self.total_reward += self.current_reward
    old_game_state_index, permutations = aux.state_to_index(old_game_state, coin_index=coin_index, dim_reduce=s.DIM_REDUCE)
    new_game_state_index, new_permutations = aux.state_to_index(new_game_state, coin_index=coin_index, dim_reduce=s.DIM_REDUCE)
    action_index = ACTION_INDICES[self_action]
    for permutation in permutations:
        action_index = PERMUTATIONS[permutation][action_index]
    # Learning according to Q learning formula on https://en.wikipedia.org/wiki/Q-learning
    self.model.Q[action_index, old_game_state_index] *= (1-ALPHA)
    self.model.Q[action_index, old_game_state_index] += ALPHA * (self.current_reward + GAMMA * max(
        self.model.Q[:, new_game_state_index]
    ))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.current_reward = 0

    coin_index = aux.index_of_closest_coin(last_game_state['self'][-1], last_game_state['coins'])

    # custom game event occurs every N_EVALSTEPS steps. If this event is active, the current distance from the coin is
    # compared to the one from the last custom event. Getting closer is rewarded, straying away is penalized.
    if last_game_state["step"] % N_EVALSTEPS == 0 and len(last_game_state["coins"]) != 0:
        self.new_distance = aux.distance_from_coin(last_game_state["self"][-1], last_game_state["coins"][coin_index])
        self.current_reward += R_DIST * (self.old_distance - self.new_distance)
        self.logger.info(f"Awarded {self.current_reward} for distance to coin")
    for event in events:
        self.current_reward += EVENT_REWARDS[event]
    self.logger.info(f"Awarded {self.current_reward} for events {', '.join(events)}")
    self.total_reward += self.current_reward
    last_game_state_index, permutations = aux.state_to_index(last_game_state, coin_index=coin_index, dim_reduce=s.DIM_REDUCE)
    action_index = ACTION_INDICES[last_action]
    for permutation in permutations:
        action_index = PERMUTATIONS[permutation][action_index]
    self.model.Q[action_index, last_game_state_index] *= (1-ALPHA)
    self.model.Q[action_index, last_game_state_index] += ALPHA * self.current_reward

    round_number = last_game_state['round']

    if round_number % 200 == 0:
        with open("training_statistic.csv", "a") as statistic_file:
            stat_writer = csv.writer(statistic_file)
            stat_writer.writerow([last_game_state['round'], last_game_state['step'], self.total_reward])

    self.total_reward = 0

    self.saving_counter += 1
    if self.saving_counter % SAVING_INTERVAL == 0:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
