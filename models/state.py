class NetworkState:
    def __init__(self):
        self.current_epoch = 0

        self.batch_size = 0
        self.epochs = 0
        self.learning_rate = 0

        self.current_training_accuracy = 0
        self.current_training_cost = 0

        self.current_validation_accuracy = 0
        self.current_validation_cost = 0

        self.test_accuracy = 0

        self.current_batch = ([], [])
