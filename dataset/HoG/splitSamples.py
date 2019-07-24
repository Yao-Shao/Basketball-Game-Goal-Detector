import numpy as np

pos = np.load("hog_pos_0.npy")
neg = np.load("hog_neg_0.npy")

trainPos = []
trainNeg = []
testPos = []
testNeg = []

cnt = 0
for i in pos:
    if cnt < len(pos)*0.7:
        trainPos.append(i)
    else:
        testPos.append(i)
    cnt += 1

cnt = 0
for i in neg:
    if cnt < len(neg)*0.7:
        trainNeg.append(i)
    else:
        testNeg.append(i)
    cnt += 1
    
np.save("train/pos_0", trainPos)
np.save("train/neg_0", trainNeg)
np.save("test/pos_0", testPos)
np.save("test/neg_0", testNeg)