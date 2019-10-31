import pickle as pkl


def load_file(path):
    with open(path, 'rb') as file:
        return pkl.load(file)


def save_file(path, data):
    with open(path, 'wb') as file:
        return pkl.dump(data, file)
