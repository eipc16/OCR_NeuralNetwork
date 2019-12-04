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

    save_plot_accuracy(data, title, f'./plots/{time_title}.png')

momentum_vs_adagrad = [
    {
        'path': './lab_3/optimizers/AdaGrad Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'AdaGrad',
        'color': 'r'
    },
    {
        'path': './lab_3/optimizers/Momentum Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Momentum',
        'color': 'y'
    },
]

adadelta_vs_adam = [
    {
        'path': './lab_3/optimizers/AdaDelta Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'AdaDelta',
        'color': 'g'
    },
    {
        'path': './lab_3/optimizers/Adam Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Adam Optimizer',
        'color': 'k'
    }
]

optimizers_plots = [
    {
        'path': './lab_3/optimizers/AdaGrad Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'AdaGrad',
        'color': 'r'
    },
    {
        'path': './lab_3/optimizers/AdaDelta Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'AdaDelta',
        'color': 'g'
    },
    {
        'path': './lab_3/optimizers/Static Gradient Descent/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Static Gradient Descent',
        'color': 'b'
    },
    {
        'path': './lab_3/optimizers/Momentum Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Momentum',
        'color': 'y'
    },
    {
        'path': './lab_3/optimizers/Adam Optimizer/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Adam Optimizer',
        'color': 'k'
    }
]

initializers = [
    {
        'path': './lab_3/initializers/xavier-gain=6/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Xavier Initializer',
        'color': 'k'
    },
    {
        'path': './lab_3/initializers/he-initializer-/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'He Initializer',
        'color': 'g'
    },
    {
        'path': './lab_3/initializers/normal-distribution-loc=0-scale=1-a=10/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Normal Initializer',
        'color': 'b'
    }
]

costs = [
    {
        'path': './lab_3/cost/func=MSE&last_layer=softmax-stable/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Softmax - MSE',
        'color': 'g'
    },
    {
        'path': './lab_3/cost/func=CrossEntropy&last_layer=sigmoid/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Sigmoid - CrossEntropy',
        'color': 'b'
    },
    {
        'path': './lab_3/cost/func=MSE&last_layer=sigmoid/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Sigmoid - MSE',
        'color': 'y'
    },
    {
        'path': './lab_3/cost/func=CrossEntropy&last_layer=softmax-stable/validation/validation_accr_batch=32_epochs=100_lr=0.pkl',
        'name': 'Softmax - CrossEntropy',
        'color': 'k'
    }
]

# load(twolayers, 'Porównanie predykcji na zbiorze walidacyjnym i treningowym')
load(momentum_vs_adagrad, 'Porównanie skuteczności Momentum i AdaGrad (predykcja na ciągu walidacyjnym)', 'momentum_vs_adagrad')
load(adadelta_vs_adam, 'Porównanie skuteczności AdaDelta i Adam (predykcja na ciągu walidacyjnym)', 'adadelta_vs_adam')
load(optimizers_plots, 'Porównanie optymalizatorów współczynnika uczenia (predykcja na ciągu walidacyjnym)', 'optimizer_plots')
load(initializers, 'Wpływ sposobu inicjacji wag na dokładność predykcji w kolejnych epokach', 'initializer_plots')
load(costs, 'Porównanie dokładności predykcji w zależności od funkcji kosztu', 'cost-function_plots')
