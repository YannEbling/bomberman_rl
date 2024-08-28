import sys
sys.path.append(".")

import os
import pickle
import numpy as np
from settings import *
from callbacks import QWalkerModel

def load(files, dir):
    matrices = []
    model = ...

    for file_name in files:
        print(file_name)

        with open(f"./agent_code/{dir}/mp/data/{file_name}", "rb") as file:
            model = pickle.load(file)
            matrices.append(model.Q)

    print(f"loaded files: {len(matrices)}")
    print(f"loaded model: {model}")
    return matrices, model

def merge(matrices):

    mats = len(matrices)
    cols = len(matrices[0])
    rows = len(matrices[0][0])

    target = np.zeros((cols, rows))

    print(f"mats {mats}")
    print(f"cols {cols}")
    print(f"rows {rows}")

    for i in range(cols):
        for j in range(rows):
            sum = 0.0
            for k in range(mats):
                sum += matrices[k][i][j]
            target[i, j] = sum / mats

    print(type(matrices))
    print(type(matrices[0]))
    print(type(matrices[0][0]))
    print(type(matrices[0][0][0]))

    #print(target)
    return target


def save(matrix, dir, model):
    with open(f"./agent_code/{dir}/my-saved-model.pt", "wb") as file:
        model.Q = matrix
        pickle.dump(model, file)
        file.close()

def list_files(directory):
    try:
        # Get a list of all files and directories in the specified directory
        all_files_and_dirs = os.listdir(directory)

        # Filter out directories, keeping only files
        files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(directory, f))]

        return files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied to access '{directory}'.")
        return []

# argv[0] = number of files to read from
def start():
    # parse argv
    argv = sys.argv[1:]
    dir = argv[0]

    # load q matrices
    matrices, model = load(list_files(f"./agent_code/{dir}/mp/data"), dir)

    # merge q matrices
    matrix = merge(matrices)

    # save merged q matrix
    save(matrix, dir, model)

start()
