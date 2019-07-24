import numpy as np
import random
import preprocess.config as config

cfg = config.Configuration()

posFiles = cfg.trainHogPos
negFiles = cfg.trainHogNeg
outTrainPos = cfg.trainSamplesPos
outTrainNeg = cfg.trainSamplesNeg
outTestPos = cfg.testSamplesPos
outTestNeg = cfg.testSamplesNeg

print("Running...")
index = 0
for file in posFiles:
    pos = np.load(file)
    trainPos = []
    testPos = []
    cnt = 0
    for i in pos:
        if cnt < len(pos) * 0.7:
            trainPos.append(i)
        else:
            testPos.append(i)
        cnt += 1
    np.save(outTrainPos[index], trainPos)
    np.save(outTestPos[index], testPos)
    index += 1

print("Process:50%")
index = 0
for file in negFiles:
    neg = np.load(file)
    trainNeg = []
    testNeg = []
    cnt = 0
    for i in neg:
        if cnt < len(neg) * 0.7:
            trainNeg.append(i)
        else:
            testNeg.append(i)
        cnt += 1
    np.save(outTrainNeg[index], trainNeg)
    np.save(outTestNeg[index], testNeg)

print("Success")