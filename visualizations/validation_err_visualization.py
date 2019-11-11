import matplotlib.pyplot as plt
import numpy as np


def show_plot(data, title=''):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_title(title)
    ax.set_xlabel('Epoka')
    ax.set_ylabel('Dokładność [%]')
    ax.grid(True)

    for x, y, name, color in data:
        ax.plot(x, y, label=name, color=color)

    ax.legend()
    plt.show()


def save_plot_training_time(data, title='', path=''):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_title(title)
    ax.set_xlabel('Test')
    ax.set_ylabel('Czas wykonania [s]')

    x, y = data

    y = np.round(y, 2)

    ax.bar(range(len(x)), y)
    plt.xticks(range(len(x)), x, rotation=40)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(path)


def save_plot_accuracy(data, title='', path=''):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_title(title)
    ax.set_xlabel('Epoka')
    ax.set_ylabel('Dokładność [%]')
    ax.grid(True)

    for x, y, name, color in data:
        ax.plot(x, y, label=name, color=color)

    ax.legend(loc='lower right')
    plt.savefig(path)

