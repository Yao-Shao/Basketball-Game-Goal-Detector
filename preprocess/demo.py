import cv2
import numpy as np


class PreparingData(object):
    def __init__(self, fn_video):
        self.fn_video = fn_video
        self.frame = ''
        self.cap = cv2.VideoCapture(fn_video)
        if not self.cap.isOpened():
            print('cannot open video file!')
            exit(-1)
        self.rect_x = ''
        self.rect_y = ''
        self.rect_width = ''
        self.rect_height = ''

    def broadcast(self, num, win_name):
        i = num
        while i > 0:
            ret, self.frame = self.cap.read()
            cv2.imshow(win_name, self.frame)
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

    def pre_process(self):
        self.broadcast(10, 'frame')

        cv2.setMouseCallback('frame', self.draw_area)
        while True:
            cv2.imshow('frame', self.frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    demo = PreparingData('1.avi')
    demo.pre_process()
