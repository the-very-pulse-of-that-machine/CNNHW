import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.applications.resnet import ResNet101
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import pathlib
from keras.layers import Input,Dense,Dropout
from keras.layers import Conv2D,MaxPool2D,GlobalAvgPool2D

import random

# 随机选择几张图片进行预测，并显示原标签和识别值
def display_predictions(val_ds, model, class_names, num_images=10):
    # 获取验证集中的所有图片和标签
    all_images, all_labels = [], []
    for images, labels in val_ds:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 随机选择图片索引
    indices = random.sample(range(len(all_images)), num_images)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        img = all_images[idx]
        label = all_labels[idx]
        # 模型预测
        prediction = model.predict(tf.expand_dims(img, axis=0))
        predicted_label = np.argmax(prediction, axis=1)[0]
        
        # 显示图片和预测结果
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img.astype("uint8"))
        plt.title(f"True: {class_names[label]}\nPred: {class_names[predicted_label]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# 调用函数显示随机图片及其标签与预测

# 设置数据目录路径
data_dir = pathlib.Path('./SID/Gray')

# 参数设置
batch_size = 32
img_height = 180
img_width = 180

# 加载训练集和验证集
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,)

# 类别名称
class_names = train_ds.class_names
print(f"类别名称: {class_names}")

# 数据预处理：缓存和预取以加快训练速度
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
pre_trained_model = ResNet101(input_shape = (180, 180, 3), # 输入大小
                                include_top = False, # 不要最后的全连接层
                                weights = 'imagenet')

x = layers.Flatten()(pre_trained_model.output)
# 加入全连接层，这个需要重头训练的
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
# 输出层
x = layers.Dense(len(class_names),activation = 'softmax')(x)  

model = Model(pre_trained_model.input, x) 

# 打印模型结构
model.summary()

# 编译模型
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

display_predictions(val_ds, model, class_names, num_images=10)

# 绘制训练曲线
plt.figure(figsize=(12, 6))

# 精度曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# 测试集评估
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(f"Validation Accuracy (as test set): {test_acc}")

# 保存模型
model.save('1.h5')
print("模型已保存！")


# 获取预测结果
y_true = []
y_pred = []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)

plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()

# 确保模型已调用
_ = model(tf.random.normal([1, img_height, img_width, 3]))  # Dummy input to define model's input shape

# 创建特征提取模型
feature_extractor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# 提取特征并进行 t-SNE 可视化
features = []
labels = []
for images, lbls in val_ds:
    features.append(feature_extractor.predict(images))
    labels.extend(lbls.numpy())
features = np.concatenate(features, axis=0)
labels = np.array(labels)

# 使用 t-SNE 降维到 2D
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features)

# 绘制 t-SNE 可视化
plt.figure(figsize=(8, 8))
for i, class_name in enumerate(class_names):
    idxs = labels == i
    plt.scatter(tsne_features[idxs, 0], tsne_features[idxs, 1], label=class_name, alpha=0.7)
plt.legend()
plt.title('t-SNE Visualization of Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
