import numpy as np
from model import NeuralNetwork

def train_model(config, X_train, y_train, X_val, y_val, save_path=None):
    model = NeuralNetwork(
        input_size=config['input_size'],
        hidden_size1=config['hidden_size1'],
        hidden_size2=config['hidden_size2'],
        output_size=config['output_size'],
        activation=config['activation']
    )
    
    learning_rate = config['learning_rate']
    decay = 1e-5  # 学习率衰减率
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        # 打乱数据
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # 小批量训练
        for i in range(0, X_train.shape[0], config['batch_size']):
            X_batch = X_train_shuffled[i:i+config['batch_size']]
            y_batch = y_train_shuffled[i:i+config['batch_size']]
            
            probs, a, z = model.forward(X_batch)
            loss = model.compute_loss(X_batch, y_batch, config['reg_strength'])
            
            # 反向传播
            grads = model.backward(X_batch, y_batch, probs, a, z, config['reg_strength'])
            
            # 更新参数
            model.weights[0] -= learning_rate * grads['W1']
            model.biases[0] -= learning_rate * grads['b1']
            model.weights[1] -= learning_rate * grads['W2']
            model.biases[1] -= learning_rate * grads['b2']
            model.weights[2] -= learning_rate * grads['W3']
            model.biases[2] -= learning_rate * grads['b3']
        
        # 学习率衰减
        learning_rate *= 1. / (1. + decay * epoch)
        
        # 验证集评估
        val_loss, val_acc = model.evaluate(X_val, y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        train_losses.append(loss)  # 记录每个epoch的最终batch loss
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Val Acc: {val_acc:.4f}")
    
    if save_path:
        model.save_weights(save_path)
    return train_losses, val_losses, val_accuracies