import os
import numpy as np
from config import DEFAULT_CONFIG, PARAM_GRID
from train import train_model
from test import test_model
from model import NeuralNetwork
from utils import (
    load_cifar10,
    normalize_data,
    split_data,
    visualize_weights,
    plot_curves
)

def main():
    # 创建权重目录
    os.makedirs('weights', exist_ok=True)
    
    # 加载并预处理数据
    X_train_full, y_train_full, X_test, y_test = load_cifar10()
    X_train_full = normalize_data(X_train_full)
    X_test = normalize_data(X_test)
    
    # 划分训练集和验证集（用于重新训练最优模型）
    X_train, y_train, X_val, y_val = split_data(X_train_full, y_train_full)
    
    # 加载超参数调优结果
    results = np.load('hyper_results.npy', allow_pickle=True).tolist()
    
    # 打印所有模型的参数和验证集准确率
    print("所有模型的超参数和验证集准确率：")
    for idx, res in enumerate(results):
        print(f"模型 {idx+1}:")
        for key, value in res.items():
            print(f"  {key}: {value}")
        print("--------------------")
    
    # 选择最优模型
    best_model = max(results, key=lambda x: x['val_acc'])
    best_params = best_model.copy()
    
    # 输出最优模型参数
    print("\n=== 最优模型参数 ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("====================")
    
    # 构造最优模型的路径
    model_path = (
        f"weights/model_"
        f"{best_params['hidden_size1']}_"
        f"{best_params['hidden_size2']}_"
        f"{best_params['learning_rate']:.1e}_"
        f"{best_params['reg_strength']:.1e}.npz"
    )
    
    # 加载最优模型并测试测试集准确率
    test_acc = test_model(model_path)
    if test_acc is not None:
        print(f"\n测试集准确率: {test_acc:.4f}")
    else:
        print("测试集准确率计算失败，模型可能未正确加载。")
    
    # =====================================================
    # 新增：重新训练最优模型以获取完整的训练过程数据
    # =====================================================
    print("\n开始重新训练最优模型以记录训练过程数据...")
    
    # 更新最优模型的配置（确保包含默认的num_epochs）
    best_params.update(DEFAULT_CONFIG)  # 合并默认配置
    
    # 调用train_model并传递config参数
    train_losses, val_losses, val_accuracies = train_model(
        config=best_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        save_path='best_model_final.npz'
    )
    
    # 保存训练过程数据
    np.save('best_model_training_logs.npy', {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    })
    
    # 可视化训练过程
    plot_curves(
        train_losses,
        val_losses,
        val_accuracies,
        title="最优模型训练过程"
    )
    
    # 可视化权重
    model = NeuralNetwork(
        input_size=DEFAULT_CONFIG['input_size'],
        hidden_size1=best_params['hidden_size1'],
        hidden_size2=best_params['hidden_size2'],
        output_size=DEFAULT_CONFIG['output_size'],
        activation='relu'
    )
    model.load_weights(model_path)
    visualize_weights(model, layer=0)  # 可视化第一层权重

if __name__ == "__main__":
    main()