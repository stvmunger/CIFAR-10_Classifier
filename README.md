# CIFAR-10 图像分类器

## 项目简介
基于神经网络对 CIFAR-10 数据集进行图像分类，支持超参数搜索和模型可视化。

## 目录结构
CIFAR-10_Classifier/

├── cifar-10-batches-py          # 数据文件夹

├── config.py          # 超参数配置

├── model.py           # 神经网络模型

├── utils.py           # 工具函数（数据加载、可视化等）

├── train.py           # 模型训练脚本

├── hyper_tuning.py    # 超参数搜索

├── test.py            # 模型测试脚本

├── main.py            # 主程序入口

├── weights/           # 保存的模型权重

└── README.md          # 项目说明

## 运行步骤
按照上述目录结构中的顺序执行各个.py文件。

1. 下载 CIFAR-10 数据集（已下载好，放在 cifar-10-batches-py 文件夹中）
2. 运行完整流程（训练+测试）python main.py
   
    （在这之前按照顺序执行 config.py, model.py, utils.py, train.py, hyper_tuning.py, test.py）
 
     输出结果：最优模型测试集准确率，训练过程曲线图（training_curves.png），第一层权重可视化图（hidden_layer_0_weights.png）

3. 超参数搜索结果

    运行 hyper_tuning.py 后，可通过以下代码查看结果：

    python


        import numpy as np
        results = np.load('hyper_results.npy', allow_pickle=True)
        for res in results:
            print(f"hidden_size1: {res['hidden_size1']}, hidden_size2: {res['hidden_size2']},"
                f"LR: {res['learning_rate']}, Reg: {res['reg_strength']},"
                f"Val Acc: {res['val_acc']:.4f}")

4. 查看最佳模型
    
    最终模型保存为 best_model_final.npz
    
    训练日志保存为 best_model_training_logs.npy



