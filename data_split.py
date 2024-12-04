import os
import shutil
import random

# 设置数据路径和分割比例
data_dir = 'SID/Gray' 
train_ratio = 0.8
test_ratio = 0.2


output_dir = 'splited'
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

for subdir in [train_dir, test_dir]:
    os.makedirs(subdir, exist_ok=True)


satellite_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]


for folder in satellite_folders:
    satellite_path = os.path.join(data_dir, folder)
    images = [f for f in os.listdir(satellite_path) if f.endswith('.bmp')]
    random.shuffle(images)  


    class_train_dir = os.path.join(train_dir, folder)
    class_test_dir = os.path.join(test_dir, folder)
    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_test_dir, exist_ok=True)


    train_count = int(train_ratio * len(images))
    

    train_images = images[:train_count]
    test_images = images[train_count:]


    for image in train_images:
        src_path = os.path.join(satellite_path, image)
        dst_path = os.path.join(class_train_dir, image)
        shutil.copy(src_path, dst_path)

    for image in test_images:
        src_path = os.path.join(satellite_path, image)
        dst_path = os.path.join(class_test_dir, image)
        shutil.copy(src_path, dst_path)

print("数据集划分完成")

