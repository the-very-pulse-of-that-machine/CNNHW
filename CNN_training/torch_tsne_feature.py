import torch

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import numpy as np



def plot_tsne_from_weights(weight_file, layer_name=None, perplexity=30, n_iter=1000):

    """

    绘制 t-SNE 特征图

    :param weight_file: str, 模型权重文件路径

    :param layer_name: str, 指定提取特征的层名（可选）

    :param perplexity: int, t-SNE 的困惑度参数

    :param n_iter: int, t-SNE 的迭代次数

    """

    # 加载模型

    model = torch.load(weight_file, map_location='cpu')

    

    # 检查模型结构

    if 'model' in model:  # 适配 YOLO 或其他模型

        state_dict = model['model'].state_dict()

    elif isinstance(model, dict) and 'state_dict' in model:

        state_dict = model['state_dict']

    else:

        state_dict = model



    # 打印模型层次

    print("模型层结构：")

    for key in state_dict.keys():

        print(f"  - {key}")



    # 提取指定层或默认层的权重

    if layer_name is None:

        # 自动选择最后一层全连接层

        layer_name = next((key for key in state_dict.keys() if 'fc' in key or 'classifier' in key), None)

        if layer_name is None:

            raise ValueError("未找到全连接层或分类层，请手动指定层名称。")

    print(f"选择的层为: {layer_name}")



    # 提取权重数据
    weights = state_dict[layer_name].detach().numpy()
    print(f"提取权重形状: {weights.shape}")

    # 检查并调整数据形状
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)  # 将 1D 数组变为 2D 数组
        print(f"调整后的权重形状: {weights.shape}")

    # 验证 perplexity 参数
    if perplexity >= weights.shape[0]:
        perplexity = max(5, weights.shape[0] // 2)  # 动态调整 perplexity
        print(f"perplexity 参数调整为: {perplexity}")

    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(weights)





    # 绘制图像

    plt.figure(figsize=(8, 6))

    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', alpha=0.6, edgecolor='k', s=50)

    plt.title(f"t-SNE Visualization of Layer: {layer_name}", fontsize=14)

    plt.xlabel("t-SNE Dimension 1")

    plt.ylabel("t-SNE Dimension 2")

    plt.grid(True)

    plt.show()



# 示例调用

weight_file = "best.pt"  # 替换为实际权重文件路径

plot_tsne_from_weights(weight_file, layer_name='model.10.linear.bias', perplexity=1400, n_iter=500)


