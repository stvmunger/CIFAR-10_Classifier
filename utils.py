import numpy as np
import os

def load_cifar10(data_dir='cifar-10-batches-py'):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')
    X_train = []
    y_train = []
    for i in range(1, 6):
        data_dict = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        X_train.append(data_dict[b'data'])
        y_train.extend(data_dict[b'labels'])
    X_train = np.concatenate(X_train).astype(np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    test_data = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test = test_data[b'data'].astype(np.float32)
    y_test = np.array(test_data[b'labels'], dtype=np.int32)
    return X_train, y_train, X_test, y_test

def normalize_data(X):
    return X / 255.0

def split_data(X, y, val_ratio=0.1):
    split_idx = int(X.shape[0] * (1 - val_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def plot_curves(train_losses, val_losses, val_accuracies, title="训练过程"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    # 绘制Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"{title} - Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title(f"{title} - Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def visualize_weights(model, layer=0):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    # 可视化指定层的权重（默认为第一层）
    weights = model.weights[layer]
    # 假设输入层到隐藏层的权重形状为 (3072, hidden_size)
    for i in range(10):  # 可视化前10个神经元的权重
        weight = weights[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
        min_val = np.min(weight)
        max_val = np.max(weight)
        normalized = (weight - min_val) / (max_val - min_val + 1e-8)
        plt.subplot(2, 5, i + 1)
        plt.imshow(normalized)
        plt.axis('off')
    plt.savefig(f'hidden_layer_{layer}_weights.png')  # 根据层号保存不同文件名
    plt.close()