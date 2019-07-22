import cv2


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


if __name__ == '__main__':
    demo = PreparingData('D:\\1.avi')
    demo.pre_process()
    demo.store_reference()
