import tensorflow as tf
import os
import cv2
import numpy as np
import random

def augment_image(image, seed):
    image_arr = []
    image_arr.append(image)
    for i in range(10):
        bright_image = tf.image.random_brightness(image, max_delta=0.4, seed=seed)
        image_arr.append(bright_image)

    for i in range(1):
        hue_image = tf.image.random_hue(image, max_delta=0.1, seed=seed)
        image_arr.append(hue_image)

    for i in range(1):
        contrast_image = tf.image.random_contrast(image, lower=0.8, upper=1.2, seed=seed)
        image_arr.append(contrast_image)

    for i in range(1):
        saturation_image = tf.image.random_saturation(image, lower=0.8, upper=1.2, seed=seed)
        image_arr.append(saturation_image)

    return np.array(image_arr)

try:
    os.mkdir('./AugmentedDataset_new/')
except:
    pass
try:
    os.mkdir('./AugmentedDataset_new/images/')
    os.mkdir('./AugmentedDataset_new/images/train/')
    os.mkdir('./AugmentedDataset_new/labels/')
    os.mkdir('./AugmentedDataset_new/labels/train/')
except:
    pass

for i in os.listdir('./uitcar/images/train/'):
    img = cv2.imread(f'./uitcar/images/train/{i}')
    i = i.replace('jpg', 'png')
    label = cv2.imread(f'./uitcar/labels/train/{i}')
    img_arr = augment_image(img, 2024)
    count = 0
    for j in img_arr:
        cv2.imwrite(f'./AugmentedDataset_new/images/train/{i.split(".")[0]}_{count}.jpg', j)
        cv2.imwrite(f'./AugmentedDataset_new/labels/train/{i.split(".")[0]}_{count}.png', label)
        count += 1

    img = cv2.flip(img, 1)
    label = cv2.flip(label, 1)
    img_arr = augment_image(img, 2024)
    for j in img_arr:
        cv2.imwrite(f'./AugmentedDataset_new/images/train/{i.split(".")[0]}_{count}.jpg', j)
        cv2.imwrite(f'./AugmentedDataset_new/labels/train/{i.split(".")[0]}_{count}.png', label)
        count += 1

try:
    os.mkdir('./AugmentedDataset_new/images/val/')
    os.mkdir('./AugmentedDataset_new/labels/val/')

except:
    pass
for i in os.listdir('./uitcar/images/val/'):
    img = cv2.imread(f'./uitcar/images/val/{i}')
    cv2.imwrite(f'./AugmentedDataset_new/images/val/{i}', img)
    i = i.replace('jpg', 'png')
    label = cv2.imread(f'./uitcar/labels/val/{i}')
    cv2.imwrite(f'./AugmentedDataset_new/labels/val/{i}', label)