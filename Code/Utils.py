import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_images(imgs, titles):
    imgs = imgs.copy()
    max_width = 0
    total_height = 0
    for img in imgs:
        h, w, _ = img.shape
        print(np.min(img))
        total_height += h
        if w > max_width:
            max_width = w
    img_composite = np.zeros((total_height, max_width, 3))
    previous_height_index = 0
    plt.figure()
    for img in imgs:
        img_composite[previous_height_index:(previous_height_index+img.shape[0]), :img.shape[1], :3] = img
        previous_height_index += img.shape[0]
    plt.imshow(cv2.cvtColor(img_composite.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.show()