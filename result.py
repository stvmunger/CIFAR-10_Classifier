import numpy as np
results = np.load('hyper_results.npy', allow_pickle=True)
for result in results:
    print(f"hidden_size1: {result['hidden_size1']}, hidden_size2: {result['hidden_size2']},learning_rate: {result['learning_rate']}, reg_strength: {result['reg_strength']}, val_acc: {result['val_acc']}")


