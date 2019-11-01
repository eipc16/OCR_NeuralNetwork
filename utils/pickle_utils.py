import os
import pickle as pkl


def load_file(path):
    with open(path, 'rb') as file:
        return pkl.load(file)


def save_file(path, file_name, data):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/{file_name}', 'wb') as file:
        return pkl.dump(data, file)
