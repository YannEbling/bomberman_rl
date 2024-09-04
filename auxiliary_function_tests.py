import numpy

import auxiliary_functions as aux
import numpy as np
import time

DIM_REDUCE = True
N = 16

# In the grids, the indexing has to be taken care of: GRID_AGENT[i][j] is corresponding to agent position (x,y), but in
# the game, the x axis is horizontal and y axis is vertical. Here it is reversed.

GRID_AGENT = [[0,  0,  0,  0,  0,  0,  0,  0,  0],
              [0,  1,  9, 13, 21, 25, 33, 37, 45],
              [0,  2,  0, 14,  0, 26,  0, 38,  0],
              [0,  3, 10, 15, 22, 27, 34, 39, 46],
              [0,  4,  0, 16,  0, 28,  0, 40,  0],
              [0,  5, 11, 17, 23, 29, 35, 41, 47],
              [0,  6,  0, 18,  0, 30,  0, 42,  0],
              [0,  7, 12, 19, 24, 31, 36, 43, 48],
              [0,  8,  0, 20,  0, 32,  0, 44,  0]]
AGENT_TILES = 48
GRID_COIN = [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  3, 16, 23,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  4,  0, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  5, 17, 25, 36, 42,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  6,  0, 26,  0, 43,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  7, 18, 27, 37, 44, 53, 58,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  8,  0, 28,  0, 45,  0, 59,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  9, 19, 29, 38, 46, 54, 60, 67, 71,  0,  0,  0,  0,  0,  0,  0],
             [0, 10,  0, 30,  0, 47,  0, 61,  0, 72,  0,  0,  0,  0,  0,  0,  0],
             [0, 11, 20, 31, 39, 48, 55, 62, 68, 73, 78, 81,  0,  0,  0,  0,  0],
             [0, 12,  0, 32,  0, 49,  0, 63,  0, 74,  0, 82,  0,  0,  0,  0,  0],
             [0, 13, 21, 33, 40, 50, 56, 64, 69, 75, 79, 83, 86, 88,  0,  0,  0],
             [0, 14,  0, 34,  0, 51,  0, 65,  0, 76,  0, 84,  0, 89,  0,  0,  0],
             [0, 15, 22, 35, 41, 52, 57, 66, 70, 77, 80, 85, 87, 90, 91, 92,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
COIN_TILES = 92
GRID_BOMB = [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [0,  1, 16, 24, 39, 47, 62, 70, 85, 93,108,116,131,139,154,162,  0],
             [0,  2,  0, 25,  0, 48,  0, 71,  0, 94,  0,117,  0,140,  0,163,  0],
             [0,  3, 17, 26, 40, 49, 63, 72, 86, 95,109,118,132,141,155,164,  0],
             [0,  4,  0, 27,  0, 50,  0, 73,  0, 96,  0,119,  0,142,  0,165,  0],
             [0,  5, 18, 28, 41, 51, 64, 74, 87, 97,110,120,133,143,156,166,  0],
             [0,  6,  0, 29,  0, 52,  0, 75,  0, 98,  0,121,  0,144,  0,167,  0],
             [0,  7, 19, 30, 42, 53, 65, 76, 88, 99,111,122,134,145,157,168,  0],
             [0,  8,  0, 31,  0, 54,  0, 77,  0,100,  0,123,  0,146,  0,169,  0],
             [0,  9, 20, 32, 43, 55, 66, 78, 89,101,112,124,135,147,158,170,  0],
             [0, 10,  0, 33,  0, 56,  0, 79,  0,102,  0,125,  0,148,  0,171,  0],
             [0, 11, 21, 34, 44, 57, 67, 80, 90,103,113,126,136,149,159,172,  0],
             [0, 12,  0, 35,  0, 58,  0, 81,  0,104,  0,127,  0,150,  0,173,  0],
             [0, 13, 22, 36, 45, 59, 68, 82, 91,105,114,128,137,151,160,174,  0],
             [0, 14,  0, 37,  0, 60,  0, 83,  0,106,  0,129,  0,152,  0,175,  0],
             [0, 15, 23, 38, 46, 61, 69, 84, 92,107,115,130,138,153,161,176,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
BOMB_TILES = 176

# All possible action permutation outcomes, to test the revert_permutations function:

PERMUTATION_RESULT = {"NO_PERM": ["LEFT", "RIGHT", "UP", "DOWN", "WAIT", "BOMB"],
                      "V_PERM": ["RIGHT", "LEFT", "UP", "DOWN", "WAIT", "BOMB"],
                      "H_PERM": ["LEFT", "RIGHT", "DOWN", "UP", "WAIT", "BOMB"],
                      "D_PERM": ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"],
                      "VH_PERM": ["RIGHT", "LEFT", "DOWN", "UP", "WAIT", "BOMB"],
                      "VD_PERM": ["DOWN", "UP", "LEFT", "RIGHT", "WAIT", "BOMB"],
                      "HD_PERM": ["UP", "DOWN", "RIGHT", "LEFT", "WAIT", "BOMB"],
                      "VHD_PERM": ["DOWN", "UP", "RIGHT", "LEFT", "WAIT", "BOMB"]}

# A dictionary to conveniently access all possible permutation combinations during testing
PERMUTATION_NAMES = {"NO_PERM": [],
                     "V_PERM": ["vertical"],
                     "H_PERM": ["horizontal"],
                     "D_PERM": ["diagonal"],
                     "VH_PERM": ["vertical", "horizontal"],
                     "VD_PERM": ["vertical", "diagonal"],
                     "HD_PERM": ["horizontal", "diagonal"],
                     "VHD_PERM": ["vertical", "horizontal", "diagonal"]}


def condition(i: int, j: int, n: int) -> bool:
    con1 = i % 2 != 0 or j % 2 != 0
    con2 = i != 0 and j != 0
    con3 = i != n and j != n
    return con1 and con2 and con3


DOMAIN = np.arange(0, 17)
VALID_POSITIONS = [(i, j) for i in DOMAIN for j in DOMAIN if condition(i, j, N)]

FAIL_POSITIONS = []
INDEX_COUNT = np.arange(1, BOMB_TILES*AGENT_TILES*COIN_TILES+1)
ERRORS_OCCURRED = False


def horizontal_mirror(positions: list[tuple], n: int) -> list[tuple]:
    new_positions = []
    for i in range(len(positions)):
        position = positions[i]
        new_position = (position[0], n - position[1])
        new_positions.append(new_position)
    return new_positions


def vertical_mirror(positions: list[tuple], n: int) -> list[tuple]:
    new_positions = []
    for i in range(len(positions)):
        position = positions[i]
        new_position = (n - position[0], position[1])
        new_positions.append(new_position)
    return new_positions


def diagonal_mirror(positions: list[tuple], n: int) -> list[tuple]:
    new_positions = []
    for i in range(len(positions)):
        position = positions[i]
        new_position = (position[1], position[0])
        new_positions.append(new_position)
    return new_positions


def main():
    print("Start testing...")
    errors_occurred = False

    for i in range(len(VALID_POSITIONS)):
        print("PROGRESS: ", i, " / ", len(VALID_POSITIONS), "(", round(i/len(VALID_POSITIONS)*100, 1), " %)")
        for j in range(len(VALID_POSITIONS)):
            for k in range(len(VALID_POSITIONS)):
                agent_pos = VALID_POSITIONS[k]
                coin_pos = VALID_POSITIONS[j]
                bomb_pos = VALID_POSITIONS[i]

                permuted = False
                if DIM_REDUCE:
                    if agent_pos[1] > N//2:
                        agent_pos, coin_pos, bomb_pos = horizontal_mirror([agent_pos, coin_pos, bomb_pos], N)
                        permuted = True
                    if agent_pos[0] > N//2:
                        agent_pos, coin_pos, bomb_pos = vertical_mirror([agent_pos, coin_pos, bomb_pos], N)
                        permuted = True
                    if coin_pos[0] < coin_pos[1]:
                        agent_pos, coin_pos, bomb_pos = diagonal_mirror([agent_pos, coin_pos, bomb_pos], N)
                        permuted = True

                try:
                    a = GRID_BOMB[bomb_pos[0]][bomb_pos[1]] - 1
                    b = GRID_COIN[coin_pos[0]][coin_pos[1]] - 1
                    c = GRID_AGENT[agent_pos[0]][agent_pos[1]] - 1
                except IndexError:
                    print(f"There is an issue with the mirroring at {i, j, k}")
                    a = b = c = 0
                expected_index = a * COIN_TILES*AGENT_TILES + b * AGENT_TILES + c

                game_state = {'self': [agent_pos],
                              'coins': [coin_pos],
                              'bombs': [[bomb_pos]],
                              'field': GRID_BOMB}

                index = -1

                computed_index, permutations = aux.state_to_index(game_state=game_state,
                                                                  coin_index=index,
                                                                  bomb_index=index,
                                                                  dim_reduce=True,   # dim_reduce has already been applied
                                                                  include_bombs=True)

                errors_occurred = errors_occurred or (computed_index != expected_index)

                global INDEX_COUNT
                if INDEX_COUNT[computed_index] > 0 and not permuted:
                    INDEX_COUNT[computed_index] = 0
                elif not permuted:
                    INDEX_COUNT[computed_index] -= 1
                    FAIL_POSITIONS.append([VALID_POSITIONS[k], VALID_POSITIONS[j], VALID_POSITIONS[i]])

                if computed_index != expected_index:
                    FAIL_POSITIONS.append([VALID_POSITIONS[k], VALID_POSITIONS[j], VALID_POSITIONS[i]])

    if not errors_occurred:
        print("The function passed the position conversion test and acted as expected.")
    else:
        print("The function failed the position conversion test.")

    maximum = max(INDEX_COUNT)
    if maximum > 0:
        print("Compactness test failed. There is at least one index which is not used (non-compactness)."
              " Largest index which isn't in use is",
              maximum-1)
    minimum = min(INDEX_COUNT)
    if minimum < 0:
        print("Uniqueness Test failed. There is at least one index which is ambiguous (used at least twice). Lowest "
              "index which is most ambiguous:", np.argmin(INDEX_COUNT), "\n number of ambiguous indices: ",
              len(FAIL_POSITIONS))

    if not len(FAIL_POSITIONS) and not maximum:
        print("The function passed the uniqueness and compactness test.")
    else:
        input_string = input("Print all failed position tuples? (y/n)")
        if input_string == 'y':
            print("All fail positions are:", FAIL_POSITIONS)
        elif input_string != 'n':
            print("Invalid input, interpreted as 'n'.")

    print("Start testing inverse permutation function.")
    error_occurred = False
    for perm in PERMUTATION_NAMES:
        error_flag = False
        initial_order = PERMUTATION_RESULT["NO_PERM"]
        permutation_outcome = PERMUTATION_RESULT[perm]
        permutations = PERMUTATION_NAMES[perm]
        for i in range(len(permutation_outcome)):
            action = permutation_outcome[i]
            new_action = aux.revert_permutations(action, permutations)
            if new_action != initial_order[i]:
                error_flag = True
        if error_flag:
            print(f"The permutation {perm} has failed the permutation inversion test.")
        error_occurred = error_occurred or error_flag

    if not error_occurred:
        print("The function has passed the permutation inversion test.")

    print("Testing finished...")


if __name__ == '__main__':
    main()
