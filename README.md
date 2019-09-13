# BallDetection
Implementation of Machine learning algorithms to detect a goal in basketball games.

- Naive classifier
- Logistic regression
- SVM
- Multiple layer perceptron
- CNN

## Introduction

### Dataset

Our dataset is four basketball videos containing about 400, 000 frames.

### Annotation

Basically, we label each frame as goal(1) or not goal(0). A frame is labeled as goal if the bottom of the basketball is below the loop and the top of the ball is in the nets. 

### Preprocess

Label, cut, and randomly shuffle the frames. See details in [preprocess](https://github.com/Yao-Shao/Basketball-Game-Goal-Detector/blob/master/preprocess/PrepareData.py).

### Feature extraction

For naive classifier, LR, SVM, we use [HoG](https://dl.acm.org/citation.cfm?id=1069007)to extract features.

### Evaluation

A ROC curve is used for evaluation.
