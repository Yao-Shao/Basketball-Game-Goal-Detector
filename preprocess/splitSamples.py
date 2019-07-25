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

partition = 7
index = 0

for file in posFiles:
    pos = np.load(file)
    trainPos = []
    testPos = []
    for i in pos:
        if random.randint(0, 9) < partition:
            trainPos.append(i)
        else:
            testPos.append(i)
    np.save(outTrainPos[index], trainPos)
    np.save(outTestPos[index], testPos)
    index += 1

index = 0
for file in negFiles:
    neg = np.load(file)
    trainNeg = []
    testNeg = []
    for i in neg:
        if random.randint(0, 9) < partition:
            trainNeg.append(i)
        else:
            testNeg.append(i)
    np.save(outTrainNeg[index], trainNeg)
    np.save(outTestNeg[index], testNeg)
    index += 1

print("Success")