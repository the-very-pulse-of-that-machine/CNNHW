import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 加载模型
model = tf.keras.models.load_model('1.h5')
print("模型已加载！")

# 加载训练集以获取类别名称
# 这里假设你已经有一个训练集 `train_ds`，从中获取类别名称
# 如果你没有 `train_ds`，你需要自己手动定义类别名称
class_names = [ 'a2100', 'astrolink', 'cobe', 'dsp', 'early-bird', 'eo1', 'ers', 'esat', 'ets8', 'fengyun', 'galileo', 'glonas', 'goms', 'helios2', 'irns', 'is-601', 'minisat-1', 'radarsat-2', 'timed', 'worldview']  # 替换为你的实际类别名称

# 自定义图片输入进行预测
def predict_image(image_path, model, class_names, img_height=180, img_width=180):
    # 加载并预处理图片
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)  # 转换为 NumPy 数组
    img_array = np.expand_dims(img_array, axis=0)  # 增加一个批次维度

    # 归一化处理 (如果你在训练时做了归一化的话)
    img_array = img_array / 255.0

    # 预测
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    # 输出预测结果
    print(f"预测类别: {class_names[predicted_class[0]]}")
    
    # 显示图片
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class[0]]}")
    plt.show()

# 输入图片路径进行预测
image_path = '2.bmp'  # 替换为你的图片路径
predict_image(image_path, model, class_names)
