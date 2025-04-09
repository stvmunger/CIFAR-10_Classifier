import numpy as np

# 加载模型文件
model = np.load('best_model_final.npz')

# 查看所有保存的键（参数名称）
print("保存的参数列表：", model.files)

# 打印每个参数的形状和前几个元素（示例）
for key in model.files:
    print(f"\n参数名: {key}")
    data = model[key]
    print(f"形状: {data.shape}")
    print("前 5 个元素：", data.flatten()[:5])