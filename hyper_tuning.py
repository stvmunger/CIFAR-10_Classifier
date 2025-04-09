import itertools
import numpy as np
import os
from config import PARAM_GRID, DEFAULT_CONFIG
from train import train_model
from utils import load_cifar10, normalize_data, split_data

def grid_search():
    # 创建权重目录
    os.makedirs('weights', exist_ok=True)
    
    X_train_full, y_train_full, _, _ = load_cifar10()
    X_train_full = normalize_data(X_train_full)
    X_train, y_train, X_val, y_val = split_data(X_train_full, y_train_full)
    
    results = []
    for params in itertools.product(*PARAM_GRID.values()):
        config = {
            **DEFAULT_CONFIG,
            'hidden_size1': params[0],
            'hidden_size2': params[1],
            'learning_rate': params[2],
            'reg_strength': params[3]
        }
        # 训练模型
        _, val_losses, val_accuracies = train_model(
            config,
            X_train, y_train,
            X_val, y_val,
            save_path=f"weights/model_{params[0]}_{params[1]}_{params[2]:.1e}_{params[3]:.1e}.npz"
        )
        val_acc = val_accuracies[-1] if val_accuracies else 0.0
        results.append({
            'hidden_size1': params[0],
            'hidden_size2': params[1],
            'learning_rate': params[2],
            'reg_strength': params[3],
            'val_acc': val_acc
        })
    np.save('hyper_results.npy', results)
    return results

if __name__ == "__main__":
    grid_search()