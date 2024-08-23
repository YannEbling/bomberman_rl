import pickle
import numpy as np
import settings as s

import auxiliary_functions as aux

NUM_TILES = (s.ROWS - 1)*(s.COLS - 1) - (s.ROWS // 2 - 1)*(s.COLS // 2 - 1)
EPSILON = 0.1
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
    if self.train:
        try:
            self.model = pickle.load(open(further_training_path, "rb"))
            self.logger.info("Model was loaded for further training.")
        except FileNotFoundError:
            self.logger.info("Setting new model from scratch.", )
            self.model = QWalkerModel(size_S=NUM_TILES**2)
    # if not training, the current model parameters should be loaded.
    else:
        self.logger.info("Loading model from saved state.")
        self.model = pickle.load(open("my-saved-model.pt", "rb"))  # What kind of data object does this produce?
        print("THIS IS THE TOTAL SUM OF Q MATRIX: ", sum(self.model.Q))


def act(self, game_state: dict):
    self.logger.debug(f"Agent is in position x: {game_state['self'][-1][0]} --- y: {game_state['self'][-1][1]}")
    return self.model.propose_action(game_state)


class QWalkerModel:
    def __init__(self, size_S, size_A=len(ACTIONS), read_from=None):
        if read_from is None:
            # The model is initialized with uniform probabilities (all actions have 1/4)
            self.Q = 1/size_A * np.ones((size_A, size_S))

    def propose_action(self, game_state):
        """
        At the index corresponding to the game_state we are in, an action from the ACTIONS catalogue should be chosen
        according to the probabilities described by the values at the corresponding entries of the Q matrix.
        """
        # with an epsilon probability, return a randomly chosen action
        num = np.random.random()
        if num < EPSILON:
            random_index = np.random.randint(low=0, high=5)
            return ACTIONS[random_index]
        index = aux.state_to_index(game_state, dim_reduce=False)  # how to do this? Which state is which row in Q?
        current_column = self.Q[:, index]

        '''
        weights = np.maximum(current_column, np.zeros(np.shape(current_column)))
        total_weight = sum(weights)
        if total_weight == 0:
            return ACTIONS[np.random.randint(5)]
        # draw a random number to choose an action probabilistically
        random_num = total_weight * np.random.random()
        if random_num < weights[0]:
            return ACTIONS[0]
        sum_so_far = weights[0]
        for i in range(1, len(weights)):
            sum_next_step = sum_so_far + weights[i]
            if sum_so_far <= random_num < sum_next_step:
                return ACTIONS[i]
            sum_so_far = sum_next_step
        '''
        return ACTIONS[np.argmax(current_column)]
        # raise ValueError("No proper action found. The summed probabilities of actions might not be normalized.")
