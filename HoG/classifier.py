import numpy as np
import preprocess.config as config
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, files):
        self.__threshold = config.thresholdMin
        sum = np.zeros(np.load(files[0])[0].shape, dtype = np.float)
        cnt = 0
        for file in files:
            samples = np.load(file)
            for item in samples:
                sum += item
            cnt += len(samples)
        self.__centerOfSamples = sum / cnt

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
    def __init__(self, clf):
        self.__classifier = clf
        self.__testPos = config.testHogPos
        self.__testNeg = config.testHogNeg
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
        while self.__classifier.getThreshold() < config.thresholdMax:
            self.testNegCenter(self.__testPos, True)
            self.testNegCenter(self.__testNeg, False)
            self.__calculate()
            self.__rec.append(self.recall)
            self.__far.append(self.falseAlarmRate)
            self.__classifier.setThreshold(self.__classifier.getThreshold() + config.thresholdStep)
            print(self.__rec, self.__far)
        # plot ROC graph
        plt.plot(self.__far, self.__rec)
        plt.title('ROC Curve')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Missed Positive Number')
        plt.axis([0,100,0,100])
        plt.grid(True)
        plt.show()




    def printEval(self):
        print("True Positive = {:}".format(self.__truePos))
        print("False Negtive = {:}".format(self.__falseNeg))
        print("True Negtive = {:}".format(self.__trueNeg))
        print("False Positive = {:}".format(self.__falsePos))

        self.__calculate()
        print("Accuracy = {:.2f}%".format(self.accuracy))
        print("Precision = {:.2f}%".format(self.precision))
        print("Recall = {:.2f}%".format(self.recall))
        print("False alarm rate = {:.2f}%".format(self.falseAlarmRate))


    def __calculate(self):
        self.accuracy = (self.__truePos + self.__trueNeg)/(self.__truePos+self.__trueNeg+self.__falseNeg+self.__falsePos) * 100
        self.precision = self.__truePos / (self.__truePos + self.__falsePos) * 100
        self.recall = self.__truePos / (self.__truePos + self.__falseNeg) * 100
        self.falseAlarmRate = self.__falsePos / (self.__falsePos + self.__trueNeg) * 100



config = config.Configuration()

if __name__ == '__main__':
    if config.task == 'train':
        index = 0

        ########## train the classifier ############
        ###### pos as center

        # trainPos = np.load(config.trainHogPos[index])
        # classifier = Classifier(trainPos)

        ###### neg as center
        classifier = Classifier(config.trainHogNeg)

        # test
        myTest = TestROC(classifier)
        # myTest.testNegCenter(config.testHogPos, True)
        # myTest.testNegCenter(config.testHogNeg, False)
        # myTest.printEval()
        myTest.draw()