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
        self.step = 180 / self.n_bins
        self.cell_dim = (int(self.win_size[0] / self.cell_size[0]), int(self.win_size[1] / self.cell_size[1]))
        self.block_dim = (int((self.win_size[0] - self.block_size[0]) / self.block_stride[0] + 1),
                          int((self.win_size[1] - self.block_size[1]) / self.block_stride[1] + 1))
        self.contain_x = int(self.block_size[0] / self.cell_size[0])
        self.contain_y = int(self.block_size[1] / self.cell_size[1])
        self.vector_length = self.block_dim[0] * self.block_dim[1] * self.contain_x * self.contain_y * self.n_bins
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
        # hog version
        self.display_win = (128, 128)
        self.max_length = 64
        self.centre = (64, 64)
        self.max_histogram = 64
        self.scale = 20
        self.color = 200
        self.thickness = 2
        self.line_type = cv2.LINE_AA
        self.win_scale = 4

    def load_data_set(self):
        print('Loading data set...')
        fn_positive = self.config.trainFnPos[self.fn_index]
        fn_negative = self.config.trainFnNeg[self.fn_index]
        tmp_positive = np.load(fn_positive)
        print('origin positive data set: ', tmp_positive.shape)
        self.positive = []
        for item in tmp_positive:
            tmp = cv2.resize(item, self.win_size)
            self.positive.append(tmp)

        tmp_negative = np.load(fn_negative)
        print('origin negative data set: ', tmp_negative.shape)
        self.negative = []
        for item in tmp_negative:
            tmp = cv2.resize(item, self.win_size)
            self.negative.append(tmp)

        print('positive set: ', len(self.positive), self.positive[0].shape)
        print('negative set: ', len(self.negative), self.negative[0].shape)
        print('Complete')

    def hog_test(self, img):
        deriv_aperture = -1
        win_sigma = -1
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 128
        hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.n_bins,
                                deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction,
                                nlevels)

        descriptor = hog.compute(img)
        if descriptor is None:
            descriptor = []
        else:
            descriptor = descriptor.ravel()

        '''
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
        '''
        return descriptor

    def cell_display(self, histogram):
        # histogram = [0, 0.11498562, 1.22206272, 2.40582829, 3.52780329, 0.64459814, 0.01769177, 0, 0]
        # histogram = [6, 6, 6, 6, 6, 6, 6, 6, 6]
        cell = np.zeros(self.display_win, np.uint8)
        for index in range(self.n_bins):
            angle = index * 20 * np.pi / 180
            length = self.max_length * (histogram[index] / self.max_histogram) * self.scale
            pt_start = (int(self.centre[0] - length * np.cos(angle)), int(self.centre[1] + length * np.sin(angle)))
            pt_end = (int(self.centre[0] + length * np.cos(angle)), int(self.centre[1] - length * np.sin(angle)))
            cv2.line(cell, pt_start, pt_end, self.color, self.thickness, self.line_type)
        cell = cv2.resize(cell, dsize=(self.cell_size[0] * self.win_scale, self.cell_size[1] * self.win_scale))
        # cv2.imshow('cell_test', cell)
        # cv2.waitKey(0)
        return cell

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

    def calc_cell_histogram(self, cell, angle):
        # dispaly
        # print(cell)
        # cv2.imshow('mag', cv2.resize(self.mag, dsize=None, fx=4, fy=4))
        # cv2.imshow('cell', cv2.resize(cell, dsize=None, fx=20, fy=20))
        # cv2.waitKey(0)

        histogram = np.zeros(self.n_bins, dtype=np.float64)
        angle = np.array(angle).flatten()
        angle = angle % 180
        cell = np.array(cell).flatten()
        for index in range(cell.shape[0]):
            down = int(angle[index] / self.step)
            dist_to_down = angle[index] % self.step
            up = (down + 1) % self.n_bins
            part_up = cell[index] * dist_to_down / self.step
            histogram[up] = histogram[up] + part_up
            histogram[down] = histogram[down] + cell[index] - part_up
        # print(histogram)

        return histogram

    def calc_cells(self):
        self.cells = []
        hog_display = np.zeros((self.win_size[0] * self.win_scale, self.win_size[1] * self.win_scale), dtype=np.uint8)
        for x in range(self.cell_dim[0]):
            cell_line = []
            for y in range(self.cell_dim[1]):
                index_x = x * self.cell_size[0]
                index_y = y * self.cell_size[1]
                cell_mag = self.mag[index_x:index_x + self.cell_size[0], index_y:index_y + self.cell_size[1]]
                cell_angle = self.angle[index_x:index_x + self.cell_size[0], index_y:index_y + self.cell_size[1]]
                cell_histogram = self.calc_cell_histogram(cell_mag, cell_angle)

                # display hog
                from_x = index_x * self.win_scale
                to_x = (index_x + self.cell_size[0]) * self.win_scale
                from_y = index_y * self.win_scale
                to_y = (index_y + self.cell_size[1]) * self.win_scale
                hog_display[from_x:to_x, from_y:to_y] = self.cell_display(cell_histogram)

                cell_line.append(cell_histogram)
            self.cells.append(cell_line)
        self.cells = np.array(self.cells)

        # display hog
        mag_display = cv2.resize(self.mag, dsize=None, fx=self.win_scale, fy=self.win_scale)
        # hog_display = cv2.resize(hog_display, dsize=None, fx=self.win_scale, fy=self.win_scale)
        cv2.imshow('mag', mag_display)
        cv2.imshow('hog', hog_display)
        cv2.waitKey(0)
        # print('cell shape = ', self.cells.shape)

    def block_normalize(self):

        # print('block dim = ', block_dim)
        step_x = int(self.block_stride[0] / self.cell_size[0])
        step_y = int(self.block_stride[1] / self.cell_size[1])

        index_x = 0
        index_y = 0
        img_vector = []
        for x in range(self.block_dim[0]):
            for y in range(self.block_dim[1]):
                block_vector = self.cells[index_x:index_x+self.contain_x, index_y:index_y+self.contain_y]
                """ get vector by for
                block_vector = []
                for i in range(contain_x):
                    for j in range(contain_y):
                        block_vector.append(self.cells[index_x + i, index_y + j])      
                """
                block_vector = np.array(block_vector)
                block_vector = block_vector.flatten()
                normalize = np.sqrt(np.sum(block_vector ** 2))
                block_vector = block_vector / (normalize + 1e-15)
                img_vector.append(block_vector)
                index_y = index_y + step_y
            index_y = 0
            index_x = index_x + step_x
        img_vector = np.array(img_vector)
        img_vector = img_vector.flatten()
        img_vector = img_vector.astype(np.float32)
        # self.hog_data_set.append(img_vector)
        # print('vector shape: ', img_vector.shape)
        return img_vector

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
        self.hog_positive = np.zeros((len(self.positive), self.vector_length), dtype=np.float32)
        index = 0
        print('HoG on positive data set...')
        for img in self.positive:
            img_vector = self.calc_hog(img)
            self.hog_positive[index] = img_vector
            index = index + 1
            if index % 200 == 0:
                print(index, end=' ')
            if index % 2000 == 0:
                print('')
        self.hog_positive = np.array(self.hog_positive)
        if self.hog_positive.shape[0] % 200 != 0:
            print('\nComplete: ', self.hog_positive.shape)

        self.hog_negative = np.zeros((len(self.negative), self.vector_length), dtype=np.float32)
        index = 0
        print('HoG on negative data set...')
        for img in self.negative:
            img_vector = self.calc_hog(img)
            # print(img_vector.shape)
            self.hog_negative[index] = img_vector
            index = index + 1
            if index % 200 == 0:
                print(index, end=' ')
            if index % 2000 == 0:
                print('')
        self.hog_negative = np.array(self.hog_negative)
        if self.hog_negative.shape[0] % 200 != 0:
            print('\nComplete: ', self.hog_negative.shape)
        self.save()


if __name__ == '__main__':
    video_index = int(input('video_index: '))
    demo = HoG(video_index)
    demo.hog_on_data_set()
