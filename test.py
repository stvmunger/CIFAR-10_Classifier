import numpy as np
from model import NeuralNetwork
from utils import load_cifar10, normalize_data

def test_model(model_path):
    # 从权重文件中读取参数形状
    data = np.load(model_path)
    W1 = data['W1']
    W2 = data['W2']
    W3 = data['W3']
    
    hidden_size1 = W1.shape[1]
    hidden_size2 = W2.shape[1]
    output_size = W3.shape[1]
    
    model = NeuralNetwork(
        input_size=W1.shape[0],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        output_size=output_size,
        activation='relu'
    )
    model.load_weights(model_path)
    
    _, _, X_test, y_test = load_cifar10()
    X_test = normalize_data(X_test)
    
    _, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")