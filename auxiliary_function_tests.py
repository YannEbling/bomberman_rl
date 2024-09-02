import numpy

import auxiliary_functions as aux
import numpy as np

DIM_REDUCE = True
N = 16

# In the following grid, i is corresponding to y and j to x. The index (i, j) therefore corresponds to the position
# (y, x). In the conversion, the positions have to be transposed.

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
             [0,  1,  2,  3,  6,  8, 13, 16, 23, 27, 36, 41, 52, 58, 71, 78,  0],
             [0,  0,  0,  4,  0,  9,  0, 17,  0, 28,  0, 42,  0, 59,  0, 79,  0],
             [0,  0,  0,  5,  7, 10, 14, 18, 24, 29, 37, 43, 53, 60, 72, 80,  0],
             [0,  0,  0,  0,  0, 11,  0, 19,  0, 30,  0, 44,  0, 61,  0, 81,  0],
             [0,  0,  0,  0,  0, 12, 15, 20, 25, 31, 38, 45, 54, 62, 73, 82,  0],
             [0,  0,  0,  0,  0,  0,  0, 21,  0, 32,  0, 46,  0, 63,  0, 83,  0],
             [0,  0,  0,  0,  0,  0,  0, 22, 26, 33, 39, 47, 55, 64, 74, 84,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0, 34,  0, 48,  0, 65,  0, 85,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0, 35, 40, 49, 56, 66, 75, 86,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 50,  0, 67,  0, 87,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 51, 57, 68, 76, 88,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 69,  0, 89,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 70, 77, 90,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 91,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 92,  0],
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
             [0, 15, 23, 38, 46, 61, 69, 84, 92,107,115,130,138,154,161,176,  0],
             [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
BOMB_TILES = 176


def condition(i: int, j: int, n: int) -> bool:
    con1 = i % 2 != 0 or j % 2 != 0
    con2 = i != 0 and j != 0
    con3 = i != n and j != n
    return con1 and con2 and con3


DOMAIN = np.arange(0, 17)
VALID_POSITIONS = [(i, j) for i in DOMAIN for j in DOMAIN if condition(i, j, N)]

FAIL_POSITIONS = []
ERRORS_OCCURRED = False


def horizontal_mirror(positions: list[tuple], n: int) -> list[tuple]:
    new_positions = []
    for i in range(len(positions)):
        position = positions[i]
        new_position = (n - position[0], position[1])
        new_positions.append(new_position)
    return new_positions


def vertical_mirror(positions: list[tuple], n: int) -> list[tuple]:
    new_positions = []
    for i in range(len(positions)):
        position = positions[i]
        new_position = (position[0], n - position[1])
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
    errors_occured = False

    for i in range(len(VALID_POSITIONS)):
        print("PROGRESS: ", i, " / ", len(VALID_POSITIONS), "(", round(i/len(VALID_POSITIONS)*100, 1), " %)")
        for j in range(len(VALID_POSITIONS)):
            for k in range(len(VALID_POSITIONS)):
                agent_pos = VALID_POSITIONS[i]
                coin_pos = VALID_POSITIONS[j]
                bomb_pos = VALID_POSITIONS[k]

                if agent_pos[0] > N//2:
                    agent_pos, coin_pos, bomb_pos = horizontal_mirror([agent_pos, coin_pos, bomb_pos], N)
                if agent_pos[1] > N//2:
                    agent_pos, coin_pos, bomb_pos = vertical_mirror([agent_pos, coin_pos, bomb_pos], N)
                if coin_pos[0] > coin_pos[1]:
                    agent_pos, coin_pos, bomb_pos = diagonal_mirror([agent_pos, coin_pos, bomb_pos], N)

                try:
                    # Here, the grids are indexed by the transposed of the positions: (y, x) = (i, j)
                    a = GRID_BOMB[bomb_pos[1]][bomb_pos[0]] - 1
                    b = GRID_COIN[coin_pos[1]][coin_pos[0]] - 1
                    c = GRID_AGENT[agent_pos[1]][agent_pos[0]] - 1
                except IndexError:
                    print(f"There is an issue with the mirroring at {i, j, k}")
                    a = b = c = 0
                expected_index = a * COIN_TILES*AGENT_TILES + b * AGENT_TILES + c

                game_state = {'self': [agent_pos],
                              'coins': [coin_pos],
                              'bombs': [[bomb_pos]],
                              'field': GRID_BOMB}

                index = -1

                computed_index = aux.state_to_index(game_state=game_state,
                                                    coin_index=index,
                                                    bomb_index=index,
                                                    dim_reduce=DIM_REDUCE,
                                                    include_bombs=True)

                errors_occurred = errors_occured and (computed_index == expected_index)

                if computed_index != expected_index:
                    FAIL_POSITIONS.append([VALID_POSITIONS[i], VALID_POSITIONS[j], VALID_POSITIONS[k]])

    if not errors_occurred:
        print("The function passed the position conversion test and acted as expected.")
    else:
        print("The function failed the position conversion test at the following (agent, coin, bomb)- positions")
        for fail_position in FAIL_POSITIONS:
            print(fail_position)


if __name__ == '__main__':
    main()
