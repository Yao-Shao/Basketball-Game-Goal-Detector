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
        self.gx = []
        self.gy = []
        self.mag = np.array([])
        self.angle = np.array([])
        self.fn_index = fn_index
        self.positive = []
        self.negative = []
        self.config = Configuration()
        self.load_data_set()
        self.hog_positive = []
        self.hog_negative = []
        self.cells = np.array([])

    def load_data_set(self):
        print('Loading data set...')
        fn_positive = self.config.trainFnPos[self.fn_index]
        fn_negative = self.config.trainFnNeg[self.fn_index]
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

    def calc_gradient(self, img):
        # img = self.positive[0]
        img = np.float64(img) / 255.0
        gx = cv2.Sobel(img, -1, 0, 1, ksize=1)
        gy = cv2.Sobel(img, -1, 1, 0, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # -- display
        # display_img = cv2.resize(img, dsize=None, fx=3, fy=3)
        # display_gx = cv2.resize(gx, dsize=None, fx=3, fy=3)
        # display_gy = cv2.resize(gy, dsize=None, fx=3, fy=3)
        # display_mag = cv2.resize(mag, dsize=None, fx=3, fy=3)
        # cv2.imshow('test_img', display_img)
        # cv2.imshow('test_x', display_gx)
        # cv2.imshow('test_y', display_gy)
        # cv2.imshow('test_mag', display_mag)
        # cv2.waitKey(0)
        # --

        self.gx = gx
        self.gy = gy
        self.mag = mag
        self.angle = angle
        return gx, gy, mag, angle

    def calc_cells(self):
        step = 180 / self.n_bins
        dim = (int(self.win_size[0] / self.cell_size[0]), int(self.win_size[1] / self.cell_size[1]))
        # print('cell dim = ', dim)
        self.cells = []
        for x in range(dim[0]):
            cell_line = []
            for y in range(dim[1]):
                histogram = np.zeros(self.n_bins, dtype=np.float64)
                index_x = x * self.cell_size[0]
                index_y = y * self.cell_size[1]
                for i in range(self.cell_size[0]):
                    for j in range(self.cell_size[1]):
                        mag = self.mag[index_x + i, index_y + j]
                        angle = self.angle[index_x + i, index_y + j]
                        if angle >= 180:
                            angle = angle - 180
                        angle_range_up = 0
                        while angle_range_up * step < angle:
                            angle_range_up = angle_range_up + 1
                        angle_range_down = angle_range_up - 1
                        part_down = mag * (angle_range_up * 20 - angle) / step
                        if angle_range_down < 0:
                            angle_range_down = self.n_bins - 1
                        if angle_range_up == self.n_bins:
                            angle_range_up = 0
                        histogram[angle_range_down] = histogram[angle_range_down] + part_down
                        histogram[angle_range_up] = histogram[angle_range_up] + mag - part_down
                cell_line.append(histogram)
            self.cells.append(cell_line)
        self.cells = np.array(self.cells)
        # print('cell shape = ', self.cells.shape)

    def block_normalize(self):
        block_dim = (int((self.win_size[0] - self.block_size[0])/self.block_stride[0] + 1),
                     int((self.win_size[1] - self.block_size[1])/self.block_stride[1] + 1))
        # print('block dim = ', block_dim)
        step_x = int(self.block_stride[0] / self.cell_size[0])
        step_y = int(self.block_stride[1] / self.cell_size[1])
        contain_x = int(self.block_size[0] / self.cell_size[0])
        contain_y = int(self.block_size[1] / self.cell_size[1])
        index_x = 0
        index_y = 0
        img_vector = []
        for x in range(block_dim[0]):
            for y in range(block_dim[1]):
                block_vector = []
                for i in range(contain_x):
                    for j in range(contain_y):
                        block_vector.append(self.cells[index_x + i, index_y + j])
                block_vector = np.array(block_vector)
                block_vector = block_vector.flatten()
                block_vector = block_vector / block_vector.sum()
                img_vector.append(block_vector)
                index_y = index_y + step_y
            index_y = 0
            index_x = index_x + step_x
        img_vector = np.array(img_vector)
        img_vector = img_vector.flatten()
        return img_vector
        # self.hog_data_set.append(img_vector)
        # print('vector shape: ', img_vector.shape)

    def calc_hog(self, img):
        self.calc_gradient(img)
        self.calc_cells()
        img_vector = self.block_normalize()
        return img_vector

    def save(self):
        out_positive = self.config.trainHogPos[self.fn_index]
        out_negative = self.config.trainHogNeg[self.fn_index]
        out_positive = out_positive[:-4]
        out_negative = out_negative[:-4]
        np.save(out_positive, self.hog_positive)
        np.save(out_negative, self.hog_negative)

    def hog_on_data_set(self):
        self.hog_positive = []
        index = 0
        print('HoG on positive data set...')
        for img in self.positive:
            img_vector = self.calc_hog(img)
            self.hog_positive.append(img_vector)
            index = index + 1
            if index % 200 == 0:
                print(index, end=' ')
            if index % 2000 == 0:
                print('')
        length = len(self.positive)
        if length % 200 != 0:
            print('Complete: ', length)

        self.hog_negative = []
        index = 0
        print('HoG on negative data set...')
        for img in self.negative:
            img_vector = self.calc_hog(img)
            self.hog_negative.append(img_vector)
            index = index + 1
            if index % 200 == 0:
                print(index, end='\t')
            if index % 2000 == 0:
                print('')
        length = len(self.negative)
        if length % 200 != 0:
            print('Complete: ', length)
        self.save()


if __name__ == '__main__':
    video_index = int(input('video_index: '))
    demo = HoG(video_index)
    # demo.hog_test()
    demo.hog_on_data_set()
