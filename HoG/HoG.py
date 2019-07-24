import cv2
import numpy as np
from preprocess.config import *


class HoG(object):
    def __init__(self, fn_index):
        # HoG reference
        self.win_size = (128, 128)
        self.block_size = (16, 16)
        self.block_stride = (8, 8)
        self.cell_size = (8, 8)
        self.n_bins = 9
        # other
        self.fn_index = fn_index
        self.positive = []
        self.negative = []
        self.config = Configuration()
        self.load_data_set()
        self.hog_positive = []
        self.hog_negative = []

    def load_data_set(self):
        print('Loading data set...')
        fn_positive = self.config.outDirPos[self.fn_index]
        fn_negative = self.config.outDirNeg[self.fn_index]
        tmp_positive = np.load(fn_positive)
        # print(tmp_positive.shape)
        self.positive = []
        for item in tmp_positive:
            tmp = cv2.resize(item, self.win_size)
            self.positive.append(tmp)

        tmp_negative = np.load(fn_negative)
        # print(tmp_negative.shape)
        self.negative = []
        for item in tmp_negative:
            tmp = cv2.resize(item, self.win_size)
            self.negative.append(tmp)

        print('positive set: ', len(self.positive), self.positive[0].shape)
        print('negative set: ', len(self.negative), self.negative[0].shape)
        print('Complete')

    def hog_test(self):
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        deriv_aperture = -1
        win_sigma = -1
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 128
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma,
                                histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)

        self.hog_positive = []
        for img in self.positive:
            # print(img.shape)

            descriptor = hog.compute(img)
            if descriptor is None:
                descriptor = []
            else:
                descriptor = descriptor.ravel()
            print(descriptor)
            # print(len(descriptor))
            self.hog_positive.append(descriptor)

        self.hog_negative = []
        for img in self.negative:
            # print(img.shape)
            descriptor = hog.compute(img)
            if descriptor is None:
                descriptor = []
            else:
                descriptor = descriptor.ravel()
            print(descriptor)
            # print(len(descriptor))
            self.hog_negative.append(descriptor)

        np.save('hog_pos_'+str(self.fn_index), self.positive)
        np.save('hog_neg_'+str(self.fn_index), self.negative)


if __name__ == '__main__':
    video_index = int(input('video_index: '))
    demo = HoG(video_index)
    demo.hog_test()
