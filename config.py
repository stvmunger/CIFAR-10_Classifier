DEFAULT_CONFIG = {
    'input_size': 3072,
    'output_size': 10,
    'activation': 'relu',
    'num_epochs': 50,
    'batch_size': 64,
    'hidden_size1': 256,
    'hidden_size2': 128,
    'learning_rate': 0.01,
    'reg_strength': 0.0001
}

PARAM_GRID = {
    'hidden_size1': [128, 256],
    'hidden_size2': [64, 128],
    'learning_rate': [0.01, 0.1],
    'reg_strength': [0.0001, 0.001]
}