import numpy as np
import preprocess.config as config
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, cfg):
        self.__config = cfg
        files = self.__config.trainSamplesNeg # neg as center
        self.__threshold = self.__config.thresholdMin
        mySum = np.zeros(np.load(files[0])[0].shape, dtype = np.float)
        cnt = 0
        for file in files:
            samples = np.load(file)
            for item in samples:
                mySum += item
            cnt += len(samples)
        self.__centerOfSamples = mySum / cnt

        # print("center:{}".format(self.__centerOfSamples))

    def __distance(self, testData):
        return np.linalg.norm(testData - self.__centerOfSamples)

    def classify(self, testData):
        # print(self.__distance(testData))
        return self.__distance(testData) < self.__threshold

    def getCenter(self):
        return self.__centerOfSamples

    def setThreshold(self, value):
        self.__threshold = value

    def getThreshold(self):
        return self.__threshold

class TestROC:
    def __init__(self, clf, cfg):
        self.__classifier = clf
        self.__config = cfg
        self.__testPos = self.__config.testSamplesPos
        self.__testNeg = self.__config.testSamplesNeg
        self.__truePos = 0
        self.__falsePos = 0
        self.__trueNeg = 0
        self.__falseNeg = 0
        self.__rec = [] # recall
        self.__far = [] # false alarm rate

    def testPosCenter(self, filePath, label):
        for file in filePath:
            data = np.load(file)
            if label == True:
                for item in data:
                    # print(label, end=':')
                    if self.__classifier.classify(item) == True:
                        self.__truePos += 1
                    else:
                        self.__falsePos += 1
            else:
                for item in data:
                    # print(label, end = ':')
                    if self.__classifier.classify(item) == True:
                        self.__falseNeg += 1
                    else:
                        self.__trueNeg += 1

    def testNegCenter(self, filePath, label):
        for file in filePath:
            data = np.load(file)
            if label == True:
                for item in data:
                    # print(label, end=':')
                    if self.__classifier.classify(item) == True:
                        self.__falsePos += 1
                    else:
                        self.__truePos += 1
            else:
                for item in data:
                    # print(label, end = ':')
                    if self.__classifier.classify(item) == True:
                        self.__trueNeg += 1
                    else:
                        self.__falseNeg += 1

    def draw(self):
        # compute data
        while self.__classifier.getThreshold() < self.__config.thresholdMax:
            self.testNegCenter(self.__testPos, True)
            self.testNegCenter(self.__testNeg, False)
            self.__calculate()
            self.printEval()
            self.__rec.append(100 - self.recall)
            self.__far.append(self.falseAlarmRate)
            self.__classifier.setThreshold(self.__classifier.getThreshold() + self.__config.thresholdStep)
        # plot ROC graph
        # print(self.__far)
        # print(self.__rec)
        plt.plot(self.__far, self.__rec)
        plt.title('ROC Curve')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Missed Positive Number')
        # plt.axis([0,100,0,100])
        plt.grid(True)
        plt.show()

    def printEval(self):
        print("Threshold = {:.2f}".format(self.__classifier.getThreshold()))
        print("True Positive = {:}".format(self.__truePos))
        print("False Negtive = {:}".format(self.__falseNeg))
        print("True Negtive = {:}".format(self.__trueNeg))
        print("False Positive = {:}".format(self.__falsePos))

        self.__calculate()
        print("Accuracy = {:.2f}%".format(self.accuracy))
        print("Precision = {:.2f}%".format(self.precision))
        print("Recall = {:.2f}%".format(self.recall))
        print("False alarm rate = {:.2f}%".format(self.falseAlarmRate))
        print()


    def __calculate(self):
        self.accuracy = (self.__truePos + self.__trueNeg)/(self.__truePos+self.__trueNeg+self.__falseNeg+self.__falsePos) * 100
        self.precision = self.__truePos / (self.__truePos + self.__falsePos) * 100
        self.recall = self.__truePos / (self.__truePos + self.__falseNeg) * 100
        self.falseAlarmRate = self.__falsePos / (self.__falsePos + self.__trueNeg) * 100


if __name__ == '__main__':
    cfg = config.Configuration()
    if cfg.task == 'train':

        ########## train the classifier ############
        ###### neg as center
        classifier = Classifier(cfg)

        ########## test ############################
        myTest = TestROC(classifier, cfg)
        myTest.draw()