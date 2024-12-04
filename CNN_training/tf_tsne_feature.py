import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def plot_tsne_from_tf_model(h5_file, layer_name=None, perplexity=30, n_iter=1000):
    # 加载模型
    model = tf.keras.models.load_model(h5_file)
    print(f"成功加载模型：{model.name}")

    # 列出所有层
    print("模型层列表：")
    for layer in model.layers:
        print(layer.name)

    # 提取指定层的权重
    if layer_name is None:
        layer_name = model.layers[-1].name  # 默认最后一层
    layer = model.get_layer(name=layer_name)
    weights = layer.get_weights()  # 获取权重

    if not weights:
        raise ValueError(f"层 {layer_name} 没有权重，无法进行 t-SNE 降维。")
    
    print(f"提取的层：{layer_name}，权重维度：{weights[0].shape}")
    features = weights[0]  # 取第一组权重 (通常是核权重)

    # 将权重展平以便 t-SNE 处理
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # 绘制 t-SNE 特征图
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=10, alpha=0.7)
    plt.title(f't-SNE 特征图: {layer_name}')
    plt.xlabel('t-SNE 维度 1')
    plt.ylabel('t-SNE 维度 2')
    plt.show()

# 示例调用
h5_file = '1.h5'
plot_tsne_from_tf_model(h5_file, layer_name=None, perplexity=40, n_iter=500)

