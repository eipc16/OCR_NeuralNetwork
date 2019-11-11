from utils.pickle_utils import load_file
from visualizations.validation_err_visualization import show_plot, save_plot_accuracy, save_plot_training_time


def load(test_cases, title, time_title):
    data = []
    data_names = []
    data_times = []
    for test_case in test_cases:
        pickle_data, time = load_file(test_case['path'])
        x, y = zip(*pickle_data)
        data.append((x, y, test_case['name'], test_case['color']))
        data_names.append(test_case['name'])
        data_times.append(time)

    save_plot_accuracy(data[1:], title, './plots/weights/weights_accuracy_without_zero.png')

batch_size = [
    {
        'path': 'results/batch_size/validation/validation_accr_batch=50000_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 50000',
        'color': 'b',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=2048_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 2048',
        'color': 'k',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=1024_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 1024',
        'color': 'r',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=100_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 100',
        'color': 'g',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 32',
        'color': 'm',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=1_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 1',
        'color': 'y',
    },
]

activations = [
    {
        'path': './results/activations/relu/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'ReLu',
        'color': 'b',
    },
    {
        'path': './results/activations/sigmoid/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Sigmoid',
        'color': 'r'
    }
]

layers_plots = [
    {
        'path': 'results/layers/one_layer_1_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '1 neuron',
        'color': 'b',
    },
    {
        'path': 'results/layers/one_layer_10_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '10 neuronów',
        'color': 'r',
    },
    {
        'path': 'results/layers/one_layer_28_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '28 neuronów',
        'color': 'c',
    },
    {
        'path': 'results/layers/one_layer_50_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '50 neuronów',
        'color': 'm',
    },
    {
        'path': 'results/layers/one_layer_300_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '300 neuronów',
        'color': 'y',
    },
    {
        'path': 'results/layers/two_layers_100_10_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '100 i 10 neuronów',
        'color': 'k',
    },
    {
        'path': 'results/layers/two_layers_100_50_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '100 i 50 neuronów',
        'color': 'g',
    },
]

twolayers = [
    {
        'path': 'results/layers/one_layer_5_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Ciąg walidacyjny',
        'color': 'g',
    },
    {
        'path': 'results/layers/one_layer_5_neurons/training/training_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Ciąg treningowy',
        'color': 'r',
    },
]

weights_plots = [
    {
        'path': 'results/weigh_initializer/zero-initializer/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Zero',
        'color': 'b',
    },
    {
        'path': 'results/weigh_initializer/xavier-gain=6/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Xavier - gain 6',
        'color': 'r',
    },
    {
        'path': 'results/weigh_initializer/xavier-gain=1/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Xavier - gain 1',
        'color': 'g',
    },
    {
        'path': 'results/weigh_initializer/range-low=-0.05-high=0.05/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Uniform - range(-0.05, 0.05)',
        'color': 'm',
    },
    {
        'path': 'results/weigh_initializer/range-low=-1-high=1/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Uniform - range(-1, 1)',
        'color': 'k',
    },
    {
        'path': 'results/weigh_initializer/range-low=-2-high=2/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Uniform - range(-2, 2)',
        'color': 'y',
    },
]

# load(twolayers, 'Porównanie predykcji na zbiorze walidacyjnym i treningowym')
load(weights_plots, 'Proces uczenia w zależności od inicjalizacji wag', '')
