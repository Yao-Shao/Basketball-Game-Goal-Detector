import cv2
import numpy as np
from preprocess.config import *
import random
import matplotlib.pyplot as plt


class DecideRect(object):
    def __init__(self, fn_video_index):
        self.config = Configuration()
        self.fn_video_index = fn_video_index
        self.fn_video = self.config.crop_vFn[fn_video_index]
        self.cap = cv2.VideoCapture(self.fn_video)
        if not self.cap.isOpened():
            print('cannot open video file!')
            exit(-1)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.label = np.zeros(self.frame_count, dtype=np.bool)
        self.frame = ''
        # rect
        self.rect = np.array([0, 0, 0, 0])
        self.rect_length = 48
        self.rect_flag = False
        self.tip_flag = False
        self.tmp = None

    def broadcast(self, num, win_name):
        i = num
        cv2.namedWindow(win_name)
        frame = ''
        while i > 0:
            ret, frame = self.cap.read()
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
            i = i - 1
        return frame

    def on_mouse(self, event, x, y, flags, param):
        if not self.rect_flag:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.rect[0] = x
                self.rect[1] = y
            elif (event == cv2.EVENT_MOUSEMOVE) & (flags & cv2.EVENT_FLAG_LBUTTON):
                self.frame = self.tmp.copy()
                self.rect_length = x - self.rect[0]
                self.rect[2] = self.rect[0] + self.rect_length
                self.rect[3] = self.rect[1] + self.rect_length
                if self.rect_length > 0:
                    self.display_rect()
                cv2.rectangle(self.frame, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), (255, 0, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                self.rect_flag = True
                self.tip_flag = True

    def display_rect(self):
        rect = self.frame[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2]]
        rect = cv2.resize(rect, dsize=None, fx=4, fy=4)
        cv2.imshow('rect', rect)

    def draw_rect(self):
        win_name = 'frame'
        self.frame = self.broadcast(20, win_name)
        self.tmp = self.frame.copy()
        cv2.setMouseCallback(win_name, self.on_mouse)

        while True:
            cv2.imshow(win_name, self.frame)

            if self.tip_flag:
                print(self.rect)
                print('Rectangular side length:', self.rect_length)
                print('if use these reference? (y/n): ')
                self.tip_flag = False

            ret = cv2.waitKey(20) & 0xFF
            if ret == ord('y'):
                    print('y')
                    break
            elif ret == ord('n'):
                self.frame = self.tmp.copy()
                self.rect_flag = False

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            self.display_rect()
            cv2.rectangle(self.frame, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), (255, 0, 0), 2)
            cv2.imshow(win_name, self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def decide_rect(self):
        self.draw_rect()
        np.save(self.config.crop_rect_fn[self.fn_video_index][:-4], self.rect)


class Label(object):
    def __init__(self, fn_video_index):
        self.config = Configuration()
        self.fn_video_index = fn_video_index
        self.rect = np.load(self.config.crop_rect_fn[fn_video_index])
        # cap
        self.fn_video = self.config.crop_vFn[fn_video_index]
        self.cap = cv2.VideoCapture(self.fn_video)
        if not self.cap.isOpened():
            print('cannot open video file!')
            exit(-1)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame = None
        # label
        self.label = np.zeros(self.frame_count, np.bool)
        # data set
        self.origin_x = None
        self.origin_y = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.num_bin = 1
        self.put_in_train_rate = 0.60
        self.flag = 1
        self.origin_x_fn = self.config.origin_x_fn[self.fn_video_index]
        self.origin_y_fn = self.config.origin_y_fn[self.fn_video_index]
        self.train_x_fn = self.config.train_x_fn[self.fn_video_index]
        self.train_y_fn = self.config.train_y_fn[self.fn_video_index]
        self.test_x_fn = self.config.test_x_fn[self.fn_video_index]
        self.test_y_fn = self.config.test_y_fn[self.fn_video_index]
        # hog data set
        self.hog = cv2.HOGDescriptor(self.config.hogWinSize, self.config.hogBlockSize, self.config.hogBlockStride,
                                     self.config.hogCellSize, self.config.hogNbins)
        self.hog_x = None

    def load_label(self):
        self.label = np.load(self.config.crop_vAnnFile[self.fn_video_index])
        # print(np.sum(self.label))

    def display(self):
        num_train_pos = np.sum(self.origin_y[0])
        num_test_pos = np.sum(self.origin_y[1])
        print('training positive: ', num_train_pos)
        print('training negative: ', len(self.origin_y[0]) - num_train_pos)
        print('testing positive: ', num_test_pos)
        print('testing negative: ', len(self.origin_y[1]) - num_test_pos)
        print(self.origin_y)

    def load_data_set(self):
        print(self.origin_x_fn)
        print(self.origin_y_fn)
        self.origin_x = np.load(self.origin_x_fn)
        self.origin_y = np.load(self.origin_y_fn)
        print('shape:', self.origin_x.shape[0])
        print('positive: ', np.sum(self.origin_y))

    def set_flag(self):
        ran = random.randint(1, 100)
        if ran <= self.put_in_train_rate * 100:
            self.flag = 0
        else:
            self.flag = 1

    def cutting(self, begin, end, flag=1):
        if begin < 0 or begin > end or end >= self.frame_count:
            print('error cutting range')
            return
        self.origin_x = []
        self.origin_y = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
        last_hog = None
        print('Cutting...')
        for i in range(begin, end+1):
            ret, self.frame = self.cap.read()
            cutting = self.frame[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2]]
            cutting = cv2.cvtColor(cutting, cv2.COLOR_BGR2GRAY)
            hog = self.hog_on_img(cutting)
            d_hog = np.zeros((hog.shape[0] * 2), dtype=np.float32)

            if flag == 1:
                self.origin_x.append(hog)
                self.origin_y.append(int(self.label[i]))
            elif i != begin:
                d_hog[:hog.shape[0]] = last_hog
                d_hog[hog.shape[0]:] = hog
                self.origin_x.append(d_hog)
                self.origin_y.append(int(self.label[i]))
            last_hog = hog.copy()
            if i % 5000 == 0:
                print(i, end=' ')
            if i % 50000 == 0 and i != begin and i != end:
                print('')
        print('\nCutting Complete')
        self.origin_x = np.asarray(self.origin_x)
        self.origin_y = np.asarray(self.origin_y)

        print('x shape: ', self.origin_x.shape)
        print('y shape: ', self.origin_y.shape)

        np.save(self.origin_x_fn[:-4], self.origin_x)
        np.save(self.origin_y_fn[:-4], self.origin_y)

    def display_rect(self):
        rect = self.frame[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2]]
        rect = cv2.resize(rect, dsize=None, fx=4, fy=4)
        cv2.imshow('rect', rect)

    def label_frame(self):
        win_name = 'label'
        cv2.namedWindow(win_name)
        print('frame count = ', self.frame_count)
        delay = 0

        pos = 0
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            print('delay = ', delay, ' frame pos = ', pos, ' label = ', self.label[pos])
            pos = pos + 1
            self.display_rect()
            cv2.rectangle(self.frame, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), (255, 0, 0), 2)
            cv2.imshow(win_name, self.frame)
            key = cv2.waitKey(delay) & 0xFF

            if (key == ord('q')) | (pos == self.frame_count - 1):
                break
            elif key == ord('u'):
                if delay == 0 or delay > 20:
                    delay = 20
                else:
                    delay = delay - 5
                    if delay <= 0:
                        delay = 1
            elif key == ord('d'):
                delay = 40
            elif key == ord('s'):
                delay = 0
            elif key == ord('c'):
                self.label[pos - 1] = not self.label[pos - 1]
                print('label change: ', not self.label[pos - 1], ' -> ', self.label[pos - 1])
            elif key == ord('b'):
                pos = pos - 20
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            elif key == ord('i'):
                tmp = input('change position: ')
                if not tmp.isnumeric():
                    print('error input')
                else:
                    tmp = int(tmp)
                    if tmp < 0 or tmp >= self.frame_count:
                        print('error input')
                    else:
                        pos = tmp
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        np.save(self.config.crop_vAnnFile[self.fn_video_index][:-4], self.label)

    def hog_on_img(self, img):
        descriptor = self.hog.compute(img)
        if descriptor is None:
            print('HOG error')
            descriptor = []
        else:
            descriptor = descriptor.ravel()
        return descriptor

    def divide(self):
        for i in range(20):
            per = np.random.permutation(self.origin_x.shape[0])
            self.origin_x = self.origin_x[per, :]
            self.origin_y = self.origin_y[per]
        epoch = np.arange(0, 80000, 1)
        plt.plot(epoch, self.origin_y, 'o')
        plt.show()
        # print(np.sum(self.origin_y))
        indices = np.random.choice(np.arange(self.origin_x.shape[0]), 50000, replace=False)
        mask = np.zeros(self.origin_x.shape[0], dtype=np.bool)
        mask[indices] = True

        self.train_x = self.origin_x[mask, :]
        self.train_y = self.origin_y[mask]
        self.test_x = self.origin_x[~mask, :]
        self.test_y = self.origin_y[~mask]
        print('train shape: ', self.train_x.shape)
        print('train positive: ', np.sum(self.train_y))
        print('test shape: ', self.test_x.shape)
        print('test positive: ', np.sum(self.test_y))

        np.save(self.train_x_fn[:-4], self.train_x)
        np.save(self.train_y_fn[:-4], self.train_y)
        np.save(self.test_x_fn[:-4], self.test_x)
        np.save(self.test_y_fn[:-4], self.test_y)


if __name__ == '__main__':
    video_index = int(input('Video Index: '))
    # demo = DecideRect(video_index)
    # demo.decide_rect()
    demo = Label(video_index)
    # demo.label_frame()
    # demo.load_label()
    # demo.cutting(100000, 180000, flag=2)
    demo.load_data_set()
    demo.divide()
