import cv2
import numpy as np
from kde import *

# 参数
threshold = 2.5e-7
file_path = 'gauss_pixel_value_road.npy'

if __name__ == '__main__':
    showBinResult(pixel_value=np.load(file_path), save_path='test.png', threshold=threshold)

