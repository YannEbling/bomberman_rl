import numpy as np
from typing import List
import settings as s


def convert_pos(position_tuple: tuple, n, is_a_coin=False):
    """
    I don't even know anymore
    """

    x, y = position_tuple
    conv_pos = 0
    subtractor = 0
    conv_pos += ((y-1)//2) * (n//2) + ((y-1) - (y-1)//2) * (n-1)
    if y % 2 == 0:
        conv_pos += (x - x//2)
    else:
        conv_pos += x

    if is_a_coin:
        for y_k in range(y+1):
            subtractor += y_k - 1
            if y_k % 2 == 0:
                subtractor -= (y_k-1)//2

    conv_pos -= subtractor
    return int(conv_pos)


def index_of_closest_item(agent_position: tuple, item_positions: List[tuple]):
    d_min = s.ROWS+s.COLS  # this is larger than every possible distance between agent and coin
    i_min = -1
    for i in range(len(item_positions)):
        d = distance_from_item(agent_position, item_positions[i])
        if d < d_min:
            i_min = i
            d_min = d
    if i_min < 0:
        raise ValueError("No coin can be found close to the agent.")
    return i_min


def state_to_index(game_state: dict, coin_index=None, bomb_index=None, dim_reduce=False, include_bombs=False):
    """
    This function is a bit messy and even if it would work, it would only be applicable for this very simple setup and
    can get very complicated. We should definitely look for a better feature extraction for more complex states.
    """
    arena_height, arena_width = np.shape(game_state["field"])
    assert arena_height == arena_width
    n = arena_height - 1
    assert n % 2 == 0

    agent_position = game_state["self"][-1]
    if len(game_state["coins"]) != 0:
        if coin_index is None:  # if there is no coin_index given...
            # ...find the closest coin
            coin_index = index_of_closest_item(agent_position, game_state["coins"])
        coin_position = game_state["coins"][coin_index]  # use the coins position
    else:
        coin_position = (0, 0)  # there are no coins and the game should end (in single player round)

    if include_bombs:
        if len(game_state['bombs']):
            if bomb_index is None:
                bomb_index = index_of_closest_item(agent_position,
                                                   [game_state['bombs'][k][0] for k in range(len(game_state['bombs']))])

            bomb_position = game_state['bombs'][bomb_index][0]
        else:
            bomb_position = (0, 0)

    permutations = []
    if dim_reduce:  # reduce the dimension of the Q matrix by exploiting mirror symmetry
        if agent_position[0] > n//2:
            agent_position_0 = n//2 - (agent_position[0] - (n//2))
            agent_position = (agent_position_0, agent_position[1])
            coin_position_0 = n//2 - (coin_position[0] - (n//2))
            coin_position = (coin_position_0, coin_position[1])
            if include_bombs:
                bomb_position_0 = n//2 - (bomb_position[0] - (n//2))
                bomb_position = (bomb_position_0, bomb_position[1])
            permutations.append("vertical")
        if agent_position[1] > n//2:
            agent_position_1 = n//2 - (agent_position[1] - (n//2))
            agent_position = (agent_position[0], agent_position_1)
            coin_position_1 = n//2 - (coin_position[1] - (n//2))
            coin_position = (coin_position[0], coin_position_1)
            if include_bombs:
                bomb_position_1 = n//2 - (bomb_position[1] - (n//2))
                bomb_position = (bomb_position[0], bomb_position_1)
            permutations.append("horizontal")
        if coin_position[1] > coin_position[0]:
            coin_position = (coin_position[1], coin_position[0])
            agent_position = (agent_position[1], agent_position[0])
            if include_bombs:
                bomb_position = (bomb_position[1], bomb_position[0])
            permutations.append("diagonal")

        number_of_agent_tiles = int(3*n**2/16)  # for the agent, only a part of the tiles are valid
        first_digit = convert_pos(coin_position, n, is_a_coin=True) - 1
        second_digit = convert_pos(agent_position, n//2+1) - 1

        index = first_digit * number_of_agent_tiles + second_digit

        if include_bombs:
            number_of_coin_tiles = 0
            for k in range(1, n):
                if k % 2 == 0:
                    number_of_coin_tiles += k // 2
                else:
                    number_of_coin_tiles += k

            third_digit = convert_pos(bomb_position, n) - 1
            number_of_agent_coin_states = number_of_coin_tiles*number_of_agent_tiles

            index += third_digit * number_of_agent_coin_states

        return index, permutations

    # This is only reached, if there is no dimension reduction.
    number_of_tiles = int(3/4 * n**2 - n)  # total number of tiles on the board
    first_digit = convert_pos(coin_position, n) - 1  # number of the tile where the agent is located
    second_digit = convert_pos(agent_position, n) - 1  # number of the tile where the coin is located

    index = first_digit * number_of_tiles + second_digit  # a unique integer, representing the state
    if include_bombs:
        if bomb_index is None: # if no bomb index is handed, find the index of the closest bomb
            bomb_index = index_of_closest_item(agent_position,
                                               [game_state['bombs'][k][0] for k in range(len(game_state['bombs']))])

        bomb_position = game_state['bombs'][bomb_index][0]
        third_digit = convert_pos(bomb_position, n) - 1  # number of the tile, where the bomb is located

        index += third_digit * number_of_tiles**2

    return index, permutations


def distance_from_item(agent_position: tuple, item_position: tuple):
    item_x, item_y = item_position
    self_x, self_y = agent_position
    return abs(item_x - self_x) + abs(item_y - self_y)


def revert_permutations(action, permutations):
    vert_dict = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "UP", "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    horiz_dict = {"LEFT": "LEFT", "RIGHT": "RIGHT", "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    diag_dict = {"LEFT": "UP", "RIGHT": "DOWN", "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}
    inverse_permutations = {"vertical": vert_dict, "horizontal": horiz_dict, "diagonal": diag_dict}
    new_action = action
    for k in range(len(permutations)):
        old_action = new_action
        permutation_to_invert = permutations[len(permutations) - (k+1)]
        new_action = inverse_permutations[permutation_to_invert][old_action]
    return new_action
