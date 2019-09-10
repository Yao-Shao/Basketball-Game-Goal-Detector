from svmutil import *
from svm import *
import numpy as np
import timeit
import six.moves.cPickle as pickle
from preprocess.config import *
import pyttsx3
from random import *


class SVM(object):
    def __init__(self):
        self.config = Configuration()
        self.v_index = self.config.video_index

        # filename
        self.train_x_fn = self.config.train_x_fn[self.v_index]
        self.train_y_fn = self.config.train_y_fn[self.v_index]
        self.test_x_fn = self.config.test_x_fn[self.v_index]
        self.test_y_fn = self.config.test_y_fn[self.v_index]

        # data set
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        # param
        self.param = self.config.param_poly_1

    def load(self):
        print('Load data...')
        self.train_x = np.load(self.train_x_fn)
        self.train_y = np.load(self.train_y_fn)
        # print('train shape: ', self.train_x.shape)
        # print('train positive: ', self.train_y.sum())
        self.test_x = np.load(self.test_x_fn)
        self.test_y = np.load(self.test_y_fn)
        # print('test shape: ', self.test_x.shape)
        # print('test positive: ', self.test_y.sum())

    def load_test_data(self):
        self.test_x = np.load(self.test_x_fn)
        self.test_y = np.load(self.test_y_fn)

    def train(self, x, y, param_index):
        start_time = timeit.default_timer()
        problem = svm_problem(y, x)
        param = svm_parameter(self.param[param_index] + ' -b 1')
        model = svm_train(problem, param)
        end_time = timeit.default_timer()
        print('Complete with time %.lf sec' % (end_time - start_time))
        # svm_save_model('../dataset/model/linear_' + self.param[param_index][9:], model)
        return model

    def cross_validation(self, folds):
        train_set_folds = np.array(np.array_split(self.train_x, folds))
        train_label_folds = np.array(np.array_split(self.train_y, folds))

        fp = open("../dataset/acc/linear.txt", 'w')

        acc_list = []
        for i in range(len(self.param)):
            print('index %d:================================' % i)
            print(self.param[i])
            acc_tmp_list = []
            for j in range(folds):
                print('folds %d:--------------------------------' % j)
                train_choose = [k for k in range(folds) if k != j]
                train_x_choose = np.concatenate(train_set_folds[train_choose])
                train_y_choose = np.concatenate(train_label_folds[train_choose])
                model = self.train(train_x_choose, train_y_choose, i)
                label, acc, val = svm_predict(train_label_folds[j], train_set_folds[j], model)
                acc_tmp_list.append(acc[0])
            mean_acc = np.mean(acc_tmp_list)
            print('param index =', i, ' acc =', mean_acc)
            acc_list.append(mean_acc)
            fp.write(self.param[i] + " " + str(mean_acc) + " " + str(acc_tmp_list) + '\n')
        best_index = np.argmax(acc_list)
        print('Complete ================================')
        print('Accuracy list: ', acc_list)
        print('Best param: ', self.param[best_index])
        print('Accuracy of best param: ', acc_list[best_index])

        np.save('log', np.array(acc_list))

    def train_all_model(self):
        print("training...")
        fp = open('../dataset/acc/poly.txt', 'a+')
        fp.write("***************************************************\n")
        cnt = 0
        for index in range(len(self.param)):
            print("process: {}/{}, param: {}".format(index+1, len(self.param), self.param[index]))
            start_time = timeit.default_timer()
            problem = svm_problem(self.train_y, self.train_x)
            param = svm_parameter(self.param[index] + ' -b 1')
            model = svm_train(problem, param)
            end_time = timeit.default_timer()
            print('Complete with time %.lf sec' % (end_time - start_time))
            svm_save_model(self.config.svm_model_fn[cnt], model)
            # save probility
            # print("calculate probility...")
            result = svm_predict(self.test_y, self.test_x, model, "-b 1")

            probility = result[2]
            fp.write(self.param[index] + ' ' + str(result[1]) + '\n')
            with open(self.config.svm_prob_fn[cnt], 'wb') as f:
                pickle.dump(np.array(probility), f)
            cnt += 1

    def get_probility(self):
        print("get probility...")
        models = self.config.svm_model_fn
        self.load_test_data()
        cnt = 0
        for model_path in models:
            print("process: {}/{}, model: {}".format(cnt+1, len(models), model_path))
            model = svm_load_model(model_path)
            result = svm_predict(self.test_y, self.test_x, model, "-b 1")
            probility = result[2]
            with open(self.config.svm_prob_fn[cnt], 'wb') as f:
                pickle.dump(np.array(probility), f)
            cnt += 1


    def predict(self, threshold, prob_fn):
        probility = pickle.load(open(prob_fn, 'rb'))
        return probility[:, 1] >= threshold, self.test_y

    def get_threshold_range(self, prob_fn):
        self.load_test_data()
        probility = pickle.load(open(prob_fn, 'rb'))
        return np.sort(np.unique(probility[:, 1]), axis=0, kind='quicksort')

if __name__ == '__main__':
    svm = SVM()
    svm.load()
    # svm.cross_validation(5)
    # svm.get_probility()
    svm.train_all_model()

    eng = pyttsx3.init()
    eng.say("Work done")
    eng.runAndWait()
