# -*- coding: utf-8 -*-
import shutil
import os

cat = []
dog = []
num = 50
cat_dir = "/home/xywang/code/pruning/catdog_classification/train/cats/"
dog_dir = "/home/xywang/code/pruning/catdog_classification/test/dogs/"
target_dir = "/home/xywang/code/pruning/Torch-Pruning/resnet50_catdog/quantized_image/"

for name in os.listdir(cat_dir):
    cat.append(cat_dir+name)
for name in os.listdir(dog_dir):
    dog.append(dog_dir+name)


def copy_img():
    final_list = cat[:num] + dog[:num]
    #final_list = dog[:num]
    for index,name in enumerate(final_list):
        shutil.copy(name,target_dir+str(index)+".jpg")
 
if __name__ == '__main__':
    copy_img()
