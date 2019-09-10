import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
from MNN.MLP_sy import *
from preprocess import config
import pyttsx3


class Evaluation:
    def __init__(self):
        # config
        self.cfg = config.Configuration()
        # ROC params
        self.__truePos = 0
        self.__falsePos = 0
        self.__trueNeg = 0
        self.__falseNeg = 0
        self.__rec = []  # recall
        self.__far = []  # false alarm rate

    def __calculate(self, threshold, prob_fn):
        pred, label = self.classifier.predict(threshold, prob_fn)
        add = pred + label
        self.__truePos = np.sum(add == 2)
        self.__trueNeg = np.sum(add == 0)
        sub = pred - label
        self.__falsePos = np.sum(sub == 1)
        self.__falseNeg = np.sum(sub == -1)
        # self.accuracy = (self.__truePos + self.__trueNeg) / (self.__truePos + self.__trueNeg + self.__falseNeg + self.__falsePos)
        # self.precision = self.__truePos / (self.__truePos + self.__falsePos)
        self.recall = self.__truePos / (self.__truePos + self.__falseNeg)
        self.falseAlarmRate = self.__falsePos / (self.__falsePos + self.__trueNeg)

    def compute(self):
        cnt = 0
        prob_fns = self.cfg.roc_prob
        # print(prob_fns)
        for prob_fn in prob_fns:
            print("\rprocess: {}/{}".format(cnt+1, len(prob_fns)), end='')
            # load clasiifier
            self.classifier = MlpOptimization()
            # compute data
            thresholds = self.classifier.get_threshold_range(prob_fn)
            for threshold in thresholds:
                # print("threshold: {}".format(threshold))
                self.__truePos = 0
                self.__falsePos = 0
                self.__trueNeg = 0
                self.__falseNeg = 0
                self.__calculate(threshold, prob_fn)
                # self.printEval()
                self.__rec.append(1 - self.recall)
                self.__far.append(self.falseAlarmRate)
                # if self.falseAlarmRate > 0.1 or self.recall > 0.99:
                #     self.__calculate(threshold_max)
                #     # self.printEval()
                #     self.__rec.append(1 - self.recall)
                #     self.__far.append(self.falseAlarmRate)
                #     break
            pickle.dump((self.__rec, self.__far), open(self.cfg.rocSavePath[cnt], 'wb'))
            self.__rec.clear()
            self.__far.clear()
            cnt += 1

    def draw(self):
        # lineStyle = ['r*-', 'bo-', 'y^-', 'g+--', 'kx-.', 'cs-']
        lineStyle = ['r-', 'b-', 'y-', 'g-', 'k-', 'c-', 'r--', 'b--', 'y--', 'g--', 'k--', 'c--']
        legend = self.cfg.roc_legend
        # load data
        inputFiles = self.cfg.rocSavePath
        cnt = 0
        for filePath in inputFiles:
            tmp = pickle.load(open(filePath,'rb'))
            self.__rec = tmp[0]
            self.__far = tmp[1]
            plt.semilogx(self.__far, self.__rec, lineStyle[cnt % len(lineStyle)], label=legend[cnt % len(legend)])
            cnt += 1
        # plot ROC graph
        plt.title(self.cfg.roc_title)
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Missed Positive Rate')
        plt.axis([1e-5, 1e-1, 0, 1])
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.show()

    def printEval(self):
        print("True Positive = {:}".format(self.__truePos))
        print("False Negtive = {:}".format(self.__falseNeg))
        print("True Negtive = {:}".format(self.__trueNeg))
        print("False Positive = {:}".format(self.__falsePos))

        # print("Accuracy = {:.6f}".format(self.accuracy))
        # print("Precision = {:.6f}".format(self.precision))
        print("Recall = {:.6f}".format(self.recall))
        print("False alarm rate = {:.6f}".format(self.falseAlarmRate))
        print()


if __name__ == '__main__':
    # evaluation = Evaluation()
    # evaluation.compute()
    # evaluation.draw()
    #
    eng = pyttsx3.init()
    eng.say("Huh, niu dong mei is pig")
    eng.runAndWait()