import pickle
import numpy as np
import settings as s

NUM_TILES = (s.ROWS - 1)*(s.COLS - 1) - (s.ROWS // 2 - 1)*(s.COLS // 2 - 1)
EPSILON = 0.1
further_training_path = "model-for-refining.pt"

ACTIONS = ['WAIT', 'UP', 'RIGHT', 'DOWN', 'LEFT']

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
        self.model = pickle.load(open("my-saved-model.pt", "rb")) # What kind of data object does this produce?
        print("THIS IS THE TOTAL SUM OF Q MATRIX: ", sum(self.model.Q))


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
    This function is a bit messy and even if it would work, it would only be applicable for this very simple setup and
    can get very complicated. We should definitely look for a better feature extraction for more complex states.
    """
    agent_position = game_state["self"][-1]
    if len(game_state["coins"]) != 0:
        coin_position = game_state["coins"][-1]
    else:
        coin_position = (0, 0)
    arena_height, arena_width = np.shape(game_state["field"])
    assert arena_height == arena_width
    n = arena_height - 1
    assert n % 2 == 0
    number_of_tiles = int(3/4 * n**2 - n)
    first_digit = convert_pos(agent_position, n)
    second_digit = convert_pos(coin_position, n)
    index = first_digit * number_of_tiles + second_digit
    return index


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
        index = state_to_index(game_state) # how to do this? Which state is which row in Q?
        current_column = self.Q[:, index]
        """
        weights = np.maximum(current_column, np.zeros(np.shape(current_column)))
        total_weight = sum(weights)
        if total_weight == 0:
            return ACTIONS[np.random.randint(5)]
        # draw a random number to chose an action probabilistically
        random_num = total_weight * np.random.random()
        if random_num < weights[0]:
            return ACTIONS[0]
        sum_so_far = weights[0]
        for i in range(1, len(weights)):
            sum_next_step = sum_so_far + weights[i]
            if sum_so_far <= random_num < sum_next_step:
                return ACTIONS[i]
            sum_so_far = sum_next_step
        print("THE LENGTH OF THE ARRAY IS: ", total_weight, end="\n\n\n\n")
        """
        return ACTIONS[np.argmax(current_column)]
        # raise ValueError("No proper action found. The summed probabilities of actions might not be normalized.")

