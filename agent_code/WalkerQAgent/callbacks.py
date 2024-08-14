import pickle
import numpy as np
import settings as s

NUM_TILES = (s.ROWS - 1)*(s.COLS - 1 - (s.ROWS // 2 - 1)*(s.COLS // 2 - 1))


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def setup(self):
    # if training is true, a new QWalkerModel is initialized. If training takes too long, there should also be the
    # possibility to further refine a pretrained model in training. All models should be saved  in QWalkerModel.pt
    if self.train:
        self.logger.info("Setting new model from scratch.", )
        self.model = QWalkerModel(size_S=NUM_TILES**2)
    # if not training, the current model parameters should be loaded.
    else:
        self.logger.info("Loading model from saved state.")
        self.model = QWalkerModel(size_S=NUM_TILES**2, read_from="QWalkerModel.pt")


def act(self, game_state: dict):
    return self.model.propose_action(game_state)


def convert_pos(position_tuple: tuple, n):
    """
    i dont even know anymore
    """
    x, y = position_tuple
    conv_pos = 0
    conv_pos += ((y-1)//2) * (n//2) + ((y-1) - (y-1)//2) * (n-1)
    if y % 2 == 0:
        conv_pos += (x - x//2)
    else:
        conv_pos += x
    return conv_pos


def state_to_index(game_state: dict):
    """
    This function is a bit messy and even if it works it is only applicable for this very simple setup and can get very
    complicated. We should definitely look for a better feature extraction for more complex states.
    """
    agent_position = game_state["self"][-1]
    print(np.shape(agent_position))
    coin_position = game_state["coins"][-1]
    print(np.shape(coin_position))
    arena_height, arena_width = np.shape(game_state["field"])
    assert arena_height == arena_width
    n = arena_height
    number_of_tiles = 3/4 * n**2 - n
    first_digit = convert_pos(agent_position, n)
    second_digit = convert_pos(coin_position, n)
    index = first_digit * number_of_tiles + second_digit
    return index


class QWalkerModel:
    def __init__(self, size_S, size_A=4, read_from=None):
        if read_from is None:
            # The model is initialized with uniform probabilities (all actions have 1/4)
            self.Q = 1/size_A * np.ones((size_A, size_S))
        else:
            self.Q = pickle.load(open(read_from, "rb")) # What kind of data object does this produce?

    def propose_action(self, game_state):
        """
        At the index corresponding to the game_state we are in, an action from the ACTIONS catalogue should be chosen
        according to the probabilities described by the values at the corresponding entries of the Q matrix.
        """
        index = state_to_index(game_state) # how to do this? Which state is which row in Q?
        weights = self.Q[:, index]
        # draw a random number to chose an action probabilistically
        random_num = np.random.rand(sum(weights))
        if random_num < weights[0]:
            return ACTIONS[0]
        for i in range(1, len(weights)):
            sum_so_far = sum(weights[:i-1])
            if sum_so_far <= random_num < sum_so_far + weights[i]:
                return ACTIONS[i]
        raise ValueError("No proper action found. The summed probabilities of actions might not be normalized.")

