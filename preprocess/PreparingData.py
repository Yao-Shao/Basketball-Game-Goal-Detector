import cv2
import numpy as np
from preprocess.config import *


class PreparingData(object):
    def __init__(self, fn_video_index):
        self.config = Configuration()
        # print(self.config.crop_vFn)
        self.fn_video = self.config.crop_vFn[fn_video_index]
        self.cap = cv2.VideoCapture(self.fn_video)
        if not self.cap.isOpened():
            print('cannot open video file!')
            exit(-1)
        self.frame = self.cap.read()
        self.rect_x = ''
        self.rect_y = ''
        self.rect_width = ''
        self.rect_height = ''
        self.win_name = 'frame'
        self.rect_flag = False
        self.fn_video_index = fn_video_index

    def broadcast(self, num):
        i = num
        cv2.namedWindow(self.win_name)
        while i > 0:
            ret, self.frame = self.cap.read()
            cv2.imshow(self.win_name, self.frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
            i = i - 1

    def draw_area(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rect_x = x
            self.rect_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.rect_width = x - self.rect_x
            self.rect_height = y - self.rect_y
            cv2.rectangle(self.frame, (x, y), (self.rect_x, self.rect_y), (255, 0, 0), 2)
            self.rect_flag = True

    def pre_process(self):
        self.broadcast(10)
        cv2.setMouseCallback(self.win_name, self.draw_area)
        tmp = self.frame.copy()
        while True:
            cv2.imshow(self.win_name, self.frame)

            if self.rect_flag:
                print('rect_x = ', self.rect_x)
                print('rect_y = ', self.rect_y)
                print('width = ', self.rect_width)
                print('height = ', self.rect_height)
                print('if use these reference? (y/n): ')
                self.rect_flag = False

            ret = cv2.waitKey(20) & 0xFF
            if ret == ord('q'):
                    break
            elif ret == ord('y'):
                    print('y')
                    break
            elif ret == ord('n'):
                self.frame = tmp.copy()
                self.rect_flag = False

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            cv2.rectangle(self.frame, (self.rect_x + self.rect_width, self.rect_y + self.rect_height),
                          (self.rect_x, self.rect_y), (255, 0, 0), 2)
            cv2.imshow(self.win_name, self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def store_reference(self):
        out = open('PreparingData.ref', 'w')
        print(self.rect_x, file=out)
        print(self.rect_y, file=out)
        print(self.rect_width, file=out)
        print(self.rect_height, file=out)
        out.close()


class MakeDataSet(object):
    def __init__(self, fn_video_index):
        self.config = Configuration()
        fn_refer = 'PreparingData.ref'
        file_refer = open(fn_refer, 'r')
        self.hoopPos = []
        self.hoopPos.append(int(file_refer.readline()))
        self.hoopPos.append(int(file_refer.readline()))
        self.hoopPos.append(self.hoopPos[0] + int(file_refer.readline()))
        self.hoopPos.append(self.hoopPos[1] + int(file_refer.readline()))
        print(self.hoopPos)
        self.fn_video = self.config.crop_vFn[fn_video_index]
        self.cap = cv2.VideoCapture(self.fn_video)
        self.frame = ''
        self.win_name = 'label windows'
        self.label = False
        self.negative = []
        self.positive = []
        self.positive_index = ''
        self.display_refer = 10
        self.fn_video_index = fn_video_index

    def cutting(self, is_display):
        cutting = self.frame[self.hoopPos[1]:self.hoopPos[3], self.hoopPos[0]:self.hoopPos[2]]
        cutting = cv2.cvtColor(cutting, cv2.COLOR_BGR2GRAY)

        if is_display:
            cutting_display = cv2.resize(cutting, dsize=None, fx=self.display_refer, fy=self.display_refer)
            cv2.imshow('cutting', cutting_display)
        return cutting

    def write_log(self, fn_video_index):
        file_ann = open(self.config.crop_vAnnFile[fn_video_index], 'w')
        for i in range(len(self.positive_index)):
            print(i, self.positive_index[i], file=file_ann)

    def read_log(self, fn_video_index):
        file_ann = open(self.config.crop_vAnnFile[fn_video_index], 'r')
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.positive_index = np.zeros(count, dtype=np.bool)
        for i in range(count):
            line = file_ann.readline()
            self.positive_index[i] = (line.find('True') >= 0)
            # print(i, line.find('True') >= 0)

    def on_change(self, emp):
        pass

    def make_data_set(self):
        cv2.namedWindow(self.win_name)
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(count)
        loop_flag = 0
        pos = 0
        cv2.createTrackbar('time', self.win_name, 0, count, self.on_change)
        delay = 0

        self.positive_index = np.zeros(count, dtype=np.bool)

        while True:
            '''
            if loop_flag == pos:
                loop_flag = loop_flag + 1
                cv2.setTrackbarPos('time', self.win_name, loop_flag)
            else:
                pos = cv2.getTrackbarPos('time', self.win_name)
                loop_flag = pos
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            '''
            pos = cv2.getTrackbarPos('time', self.win_name)
            if pos != loop_flag:
                loop_flag = pos
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, self.frame = self.cap.read()
            if not ret:
                break
            # next_loop_flag = loop_flag + 1
            # next_pos = pos + 1

            loop_flag = loop_flag + 1
            pos = pos + 1
            cv2.setTrackbarPos('time', self.win_name, pos)

            cutting = self.cutting(True)

            if self.label & (self.positive_index[pos - 1] == 0):
                self.positive_index[pos - 1] = 1
                self.positive.append(cutting)

            # print(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print('delay = ', delay, ' frame pos = ', pos - 1, ' label = ', self.positive_index[pos - 1])

            cv2.rectangle(self.frame, (self.hoopPos[2], self.hoopPos[3]),
                          (self.hoopPos[0], self.hoopPos[1]), (255, 0, 0), 2)
            cv2.imshow(self.win_name, self.frame)

            key = cv2.waitKey(delay) & 0xFF

            if (key == ord('q')) | (pos == count):
                break
            elif key == ord('u'):
                if delay == 0 or delay > 20:
                    delay = 20
                else:
                    delay = delay - 5
                    if delay <= 0:
                        delay = 1
            elif key == ord('d'):
                delay = 1000
            elif key == ord('s'):
                delay = 0
            elif key == ord('c'):
                self.label = not self.label
            elif key == ord('i'):
                next_pos = int(input('change position: '))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                cv2.setTrackbarPos('time', self.win_name, next_pos)

        self.cap = cv2.VideoCapture(self.fn_video)
        index = 0
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            if self.positive_index[index] == 0:
                cutting = self.cutting(False)
                self.negative.append(cutting)
            index = index + 1

    def output_data_set(self, fn_index):
        # test item in list
        # for item in self.negative:
        #    cv2.imshow('test', item)
        #    cv2.waitKey(10)

        # print(type(self.negative[0]), self.negative[0].shape)
        np.save(self.config.outDirNeg[fn_index][:-4], self.negative)
        np.save(self.config.outDirPos[fn_index][:-4], self.positive)
        print(len(self.positive))
        print(len(self.negative))

        # test save result
        # test = np.load(fn_pos+'.npy')
        # for item in test:
        #    cv2.imshow('test', item)
        #    cv2.waitKey(20)

    def make_data_set_by_log(self, fn_video_index):
        self.read_log(fn_video_index)
        self.cap = cv2.VideoCapture(self.fn_video)
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('count = ', count)
        self.positive = []
        self.negative = []
        for i in range(count):
            ret, self.frame = self.cap.read()
            cutting = self.cutting(False)
            if self.positive_index[i]:
                self.positive.append(cutting)
            else:
                self.negative.append(cutting)
            if (i+1) % 5000 == 0:
                print(i+1, end='\t')
            if (i+1) % 50000 == 0:
                print(end='\n')

        print('\nComplete')

        for positive_item in self.positive:
            display = cv2.resize(positive_item, dsize=None, fx=self.display_refer, fy=self.display_refer)
            cv2.imshow(self.win_name, display)
            cv2.waitKey(10)

    def explore(self, npy_file):
        crop = np.load(npy_file)
        for image in crop:
            display = cv2.resize(image, dsize=None, fx=self.display_refer, fy=self.display_refer)
            cv2.imshow(self.win_name, display)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    video_index = int(input('video index: '))
    op = input('operation: ')
    if op == '0':
        demo = PreparingData(video_index)
        demo.pre_process()
        demo.store_reference()
    else:
        demo = MakeDataSet(video_index)

        # demo.make_data_set()
        # demo.output_data_set(video_index)
        # demo.write_log(video_index)

        # demo.make_data_set_by_log(video_index)
        # demo.output_data_set(video_index)

        demo.explore(demo.config.outDirPos[demo.fn_video_index])
        demo.explore(demo.config.outDirNeg[demo.fn_video_index])
