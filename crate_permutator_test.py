from auxiliary_functions import crate_permutator
import numpy as np


all_permutations = [['vertical'],
                ['horizontal'],
                ['diagonal'],
                ['vertical', 'horizontal'],
                ['vertical', 'diagonal'],
                ['horizontal', 'diagonal'],
                ['vertical', 'horizontal', 'diagonal']
                ]
binary = [0, 1]
for permutations in all_permutations:
    print("\n\n______PERMUTATIONS: ", permutations, " ________\n")
    for old_up in binary:
        for old_right in binary:
            for old_down in binary:
                for old_left in binary:
                    old_grid = np.zeros((3, 3), dtype=int)
                    new_grid = np.zeros((3, 3), dtype=int)
                    a, b, c, d = crate_permutator([old_up,
                                                   old_right,
                                                   old_down,
                                                   old_left],
                                                  permutations)
                    old_grid[0, 1] = old_up
                    old_grid[1, 2] = old_right
                    old_grid[2, 1] = old_down
                    old_grid[1, 0] = old_left

                    new_grid[0, 1] = a
                    new_grid[1, 2] = b
                    new_grid[2, 1] = c
                    new_grid[1, 0] = d

                    print("\nOLD:")
                    print(old_grid)

                    print("\nNEW: ")
                    print(new_grid)
