import cv2
import numpy as np


class PreparingData(object):
    def __init__(self, fn_video):
        self.fn_video = fn_video
        self.cap = cv2.VideoCapture(fn_video)
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
    def __init__(self, fn_video):
        fn_refer = 'PreparingData.ref'
        file_refer = open(fn_refer, 'r')
        self.hoopPos = []
        self.hoopPos.append(int(file_refer.readline()))
        self.hoopPos.append(int(file_refer.readline()))
        self.hoopPos.append(self.hoopPos[0] + int(file_refer.readline()))
        self.hoopPos.append(self.hoopPos[1] + int(file_refer.readline()))
        print(self.hoopPos)
        self.cap = cv2.VideoCapture(fn_video)
        self.frame = ''
        self.win_name = 'label windows'
        self.label = False
        self.negative = []
        self.positive = []

    def cutting(self):
        cutting = self.frame[self.hoopPos[1]:self.hoopPos[3], self.hoopPos[0]:self.hoopPos[2]]
        cutting = cv2.cvtColor(cutting, cv2.COLOR_BGR2GRAY)
        if self.label:
            self.positive.append(cutting)
        else:
            self.negative.append(cutting)
        cutting_display = cv2.resize(cutting, dsize=None, fx=2, fy=2)
        cv2.imshow('cutting', cutting_display)

    def write_log(self):
        pass

    def make_data_set(self):
        delay = 20
        count = 0
        while True:
            count = count + 1
            ret, self.frame = self.cap.read()
            print('delay = ', delay, ' frame count = ', count, ' label = ', self.label)
            if not ret:
                break

            self.cutting()

            cv2.rectangle(self.frame, (self.hoopPos[2], self.hoopPos[3]),
                          (self.hoopPos[0], self.hoopPos[1]), (255, 0, 0), 2)
            cv2.imshow(self.win_name, self.frame)

            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('u'):
                if delay == 0 or delay > 20:
                    delay = 20
                else:
                    delay = delay - 5
                    if delay <= 0:
                        delay = 1
            elif key == ord('d'):
                delay = 100
            elif key == ord('s'):
                delay = 0
            elif key == ord('c'):
                self.label = not self.label

    def output_data_set(self, fn_neg, fn_pos):
        # test item in list
        # for item in self.negative:
        #    cv2.imshow('test', item)
        #    cv2.waitKey(10)

        print(type(self.negative[0]), self.negative[0].shape)
        np.save(fn_neg, self.negative)
        np.save(fn_pos, self.positive)

        # test save result
        test = np.load(fn_neg+'.npy')
        for item in test:
            cv2.imshow('test', item)
            cv2.waitKey(20)


if __name__ == '__main__':
    demo = MakeDataSet("D:\\1.avi")
    demo.make_data_set()

