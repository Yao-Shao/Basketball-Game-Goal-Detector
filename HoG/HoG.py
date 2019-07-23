import cv2
import numpy as np

fp = np.load("../preprocess/pos.npy")
for item in fp:
    print(item)
    print(type(item))

class HoG:
    def __init__(self):
        