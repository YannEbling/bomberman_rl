import pickle
import numpy as np
import settings as s
import random

import auxiliary_functions as aux

N = s.ROWS - 1
NUM_TILES = (N - 1)*(N - 1) - (N // 2 - 1)*(N // 2 - 1)
EPSILON = 0.25
further_training_path = "model-for-refining.pt"

ACTIONS = ['WAIT', 'UP', 'RIGHT', 'DOWN', 'LEFT']

ACTION_INDICES = {'WAIT': 0,
                  'UP': 1,
                  'RIGHT': 2,
                  'DOWN': 3,
                  'LEFT': 4
                  }

PERMUTATIONS = {"vertical": [0, 1, 4, 3, 2],
                "horizontal": [0, 3, 2, 1, 4],
                "diagonal": [0, 4, 3, 2, 1]
                }


def setup(self):
    # if training is true, a new QWalkerModel is initialized. If training takes too long, there should also be the
    # possibility to further refine a pretrained model in training. All models should be saved  in QWalkerModel.pt

    assert N+1 == s.COLS
    if self.train:
        try:
            self.model = pickle.load(open(further_training_path, "rb"))
            self.logger.info("Model was loaded for further training.")
        except FileNotFoundError:
            if s.DIM_REDUCE:
                number_of_agent_tiles = int(3*N**2/16)
                number_of_coin_tiles = 0
                for k in range(1, N):
                    if k % 2 == 0:
                        number_of_coin_tiles += k // 2
                    else:
                        number_of_coin_tiles += k
                size_S = number_of_agent_tiles*number_of_coin_tiles
                self.logger.info("Dimension of model is reduced exploiting mirror symmetry.")
            else:
                size_S = NUM_TILES**2
            self.logger.info("Setting new model from scratch.", )
            self.model = QWalkerModel(size_S, dim_reduce=s.DIM_REDUCE)
    # if not training, the current model parameters should be loaded.
    else:
        self.logger.info("Loading model from saved state.")
        self.model = pickle.load(open("my-saved-model.pt", "rb"))


def act(self, game_state: dict):
    action = self.model.propose_action(game_state, self.train, self.logger)
    s.MOST_RECENT_ACTION = action
    return action


class QWalkerModel:
    def __init__(self, size_S, size_A=len(ACTIONS), read_from=None, dim_reduce=False):
        self.dim_reduce = dim_reduce
        if read_from is None:
            # The model is initialized with uniform probabilities (all actions have 1/4)
            self.Q = 1/size_A * np.ones((size_A, size_S)) # size_S is the number of states, depends wether the dimension
            # is reduced or not (see setup())
            print("SIZE OF MATRIX: ", size_A, " x ", size_S)

    def propose_action(self, game_state, train, logger):
        """
        At the index corresponding to the game_state we are in, an action from the ACTIONS catalogue should be chosen
        according to the probabilities described by the values at the corresponding entries of the Q matrix.
        """

        coin_index = aux.index_of_closest_item(game_state['self'][-1], game_state['coins'])
        index, permutations = aux.state_to_index(game_state, coin_index=coin_index, dim_reduce=self.dim_reduce)

        # with an epsilon probability, return a randomly chosen action

        if train:
            num = np.random.random()
            if num < EPSILON:
                action_index = np.random.randint(low=0, high=5)
                logger.info(f"Chose action {ACTIONS[action_index]} purely at random.")
                return ACTIONS[action_index]
        current_column = self.Q[:, index]
        max_element = 0
        index_to_chose = []
        for k in range(len(current_column)):
            if current_column[k] > max_element:
                max_element = current_column[k]
                index_to_chose = [k]
            elif current_column[k] == max_element:
                index_to_chose.append(k)
        if len(index_to_chose):
            action_index = random.choice(index_to_chose)
        else:
            action_index = random.randint(0, 4)
        logger.info(f"In step {game_state['step']} the Q matrix row is {current_column}.")
        # action_index = np.argmax(current_column)
        logger.info(f"The chosen action is {ACTIONS[action_index]}.")
        for k in range(len(permutations), 0, -1):
            action_index = PERMUTATIONS[permutations[k-1]][action_index]
        logger.info(f"After undoing the permutations, the action is {ACTIONS[action_index]}")

        '''
        weights = np.maximum(current_column, np.zeros(np.shape(current_column)))
        total_weight = sum(weights)
        if total_weight == 0:
            return ACTIONS[np.random.randint(5)]
        # draw a random number to choose an action probabilistically
        random_num = total_weight * np.random.random()
        print("Returning a non random action.")
        if random_num < weights[0]:
            action_index = ACTION_INDICES[ACTIONS[0]]
            for k in range(len(permutations), 0, -1):
                action_index = PERMUTATIONS[permutations[k-1]][action_index]
            return action_index
        sum_so_far = weights[0]
        for i in range(1, len(weights)):
            sum_next_step = sum_so_far + weights[i]
            if sum_so_far <= random_num < sum_next_step:
                action_index = ACTION_INDICES[ACTIONS[i]]
                for k in range(len(permutations), 0, -1):
                    action_index = PERMUTATIONS[permutations[k-1]][action_index]
                return action_index
            sum_so_far = sum_next_step
        # raise ValueError("No proper action found. The summed probabilities of actions might not be normalized.")

        '''
        #action_index = np.argmax(current_column)
        #for k in range(len(permutations), 0, -1):
        #    action_index = PERMUTATIONS[permutations[k-1]][action_index]
        return ACTIONS[action_index]

