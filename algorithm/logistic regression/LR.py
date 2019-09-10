import numpy as np
import timeit
import six.moves.cPickle as pickle
import preprocess.config as config
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, n_in, n_out):

        self.cfg = config.Configuration()
        self.n_class = n_out
        self.W = np.zeros((n_in, n_out), dtype=np.float)
        self.b = np.zeros((n_out,), dtype=np.float)
        self.learning_rate = self.cfg.lr_learning_rate
        self.n_epochs = self.cfg.lr_n_epoch
        self.batch_size = self.cfg.lr_batch_size

        # compute p_y given x
        self.p_y_given_x = None
        self.exp_x_multiply_w_plus_b = None

        # gradient_w_b
        self.delta_W = 0
        self.delta_b = 0

        # train
        self.patience = self.cfg.lr_patience
        self.patience_increase = self.cfg.lr_patience_increase
        self.improvement_threshold = self.cfg.lr_improvement_threshold
        self.lamda = self.cfg.lr_weight_decay_lamda

        # data set
        self.train_set_x = None
        self.train_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.test_set_x = None
        self.test_set_y = None

        # params
        self.is_weight_decay = self.cfg.lr_weight_decay
        self.is_line_search = self.cfg.lr_line_search
        self.is_momentum = self.cfg.is_momentum
        self.momentum_vW = 0
        self.momentum_vb = 0

        # line search test
        self.num_test_epoch = 200
        self.test_result = np.zeros((self.num_test_epoch + 1, 4), dtype=np.float32)

    def negative_log_likelihood(self, index=-1):
        if index == -1:
            x = self.train_set_x
            y = self.train_set_y
        else:
            x = self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
            y = self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]

        exp_x_multiply_w_plus_b = np.exp(np.dot(x, self.W) + self.b)
        sigma = np.sum(exp_x_multiply_w_plus_b, axis=1)
        p_y_given_x = exp_x_multiply_w_plus_b / sigma.reshape(sigma.shape[0], 1)
        return -np.mean(np.log(p_y_given_x)[np.arange(y.shape[0]), y])

    def zero_one_errors(self, index=0, flag=1):
        self.compute_p_y_given_x(index, flag=flag)
        if flag == 1:
            y = self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        elif flag == 2:
            y = self.valid_set_y
        else:
            y = self.test_set_y
        predict_y = np.argmax(self.p_y_given_x, axis=1)
        return np.mean(predict_y != y)

    def compute_p_y_given_x(self, index=0, flag=1, j=-1):
        if flag == 1:
            x = self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        elif flag == 2:
            x = self.valid_set_x
        else:
            x = self.test_set_x

        if j == -1:
            self.exp_x_multiply_w_plus_b = np.exp(np.dot(x, self.W) + self.b)
        else:
            exp_x_multiply_w_plus_b_j = np.exp(np.dot(x, self.W[:, j]) + self.b[j])
            self.exp_x_multiply_w_plus_b[:, j] = exp_x_multiply_w_plus_b_j[:]

        sigma = np.sum(self.exp_x_multiply_w_plus_b, axis=1)
        self.p_y_given_x = self.exp_x_multiply_w_plus_b / sigma.reshape(sigma.shape[0], 1)

    def gradient_w_b(self, index):
        x = self.train_set_x[index * self.batch_size: (index+1)*self.batch_size]
        y = self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        y_is_j = (y.reshape(y.shape[0], 1)) == np.array(np.arange(self.n_class), dtype=np.int)
        coef = y_is_j - self.p_y_given_x

        self.delta_W = (-1.0 * np.dot(coef.transpose(), x) / y.shape[0]).transpose()
        self.delta_b = -1.0 * np.mean(coef, axis=0)

        if self.is_weight_decay:
            self.delta_W += self.lamda * self.W
            self.delta_b += self.lamda * self.b

        if self.is_momentum:
            self.momentum_vW = self.cfg.train_momentum_m * self.momentum_vW - self.learning_rate * self.delta_W
            self.momentum_vb = self.cfg.train_momentum_m * self.momentum_vb - self.learning_rate * self.delta_b

    def update_w_b(self, index):
        if self.is_line_search:
            self.wolfe_line_search(index)
        else:
            if self.is_momentum:
                self.W += self.momentum_vW
                self.b += self.momentum_vb
            else:
                self.W -= self.learning_rate * self.delta_W
                self.b -= self.learning_rate * self.delta_b

    def wolfe_line_search(self, index):
        i = 0
        c = 0.5
        tau = 0.5
        slope = (self.delta_W ** 2).sum(axis=0)

        while i < self.n_class:
            t_learning_rate = 1.0
            ori_loss = self.negative_log_likelihood(index)
            self.W[:, i] -= t_learning_rate * self.delta_W[:, i]
            prev_learning_rate = t_learning_rate
            while 1:
                tt = c * t_learning_rate * slope[i]
                self.compute_p_y_given_x(index, j=i)
                curr_loss = self.negative_log_likelihood(index)
                if curr_loss <= ori_loss - tt:
                    break
                else:
                    t_learning_rate *= tau
                    if t_learning_rate < self.learning_rate:
                        t_learning_rate = self.learning_rate
                        self.W[:, i] += (prev_learning_rate - t_learning_rate) * self.delta_W[:, i]
                        self.compute_p_y_given_x(index, j=i)
                        break
                self.W[:, i] += (prev_learning_rate - t_learning_rate) * self.delta_W[:, i]
                prev_learning_rate = t_learning_rate
            i += 1

    def sgd_optimization(self):

        # load train and validation data set
        self.load_data()

        # compute number of minibatches for training, validation and testing
        n_train_batches = self.train_set_x.shape[0] // self.batch_size

        validation_frequency = min(n_train_batches, self.patience//2)

        best_validation_loss = np.inf

        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        best_param = [self.W, self.b]

        print('... trianing model')
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in np.arange(n_train_batches):
                self.compute_p_y_given_x(minibatch_index)
                self.gradient_w_b(minibatch_index)
                self.update_w_b(minibatch_index)

                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    this_validation_loss = self.zero_one_errors(flag=2)
                    # train_loss = self.negative_log_likelihood()
                    # print('epoch %i, minibatch %i, left patience %d, train loss %f, validation error %f %%' % (
                    #         epoch,
                    #         minibatch_index + 1,
                    #         self.patience - iter,
                    #         train_loss,
                    #         this_validation_loss * 100))
                    print('epoch %i, minibatch %i, left patience %d, validation error %f %%' % (
                        epoch,
                        minibatch_index + 1,
                        self.patience - iter,
                        this_validation_loss * 100))
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * self.improvement_threshold:
                            self.patience = max(self.patience, iter * self.patience_increase)
                        best_validation_loss = this_validation_loss
                        best_param = [self.W, self.b]
                if self.patience <= iter:
                    done_looping = True
                    break
                if best_validation_loss < 1e-5:
                    break
        end_time = timeit.default_timer()
        print("Optimization complete with best validation loss of %f %%" % (best_validation_loss * 100))
        print("The code run for %d epochs, with %f epoch/sec, total time %.1f sec" %
              (epoch, 1.0 * epoch / (end_time - start_time), (end_time - start_time)))
        # save best model
        print('... save best model')
        with open(self.cfg.train_fn_model[0], 'wb') as f:
            pickle.dump(best_param, f)

    def get_threshold_range(self):
        self.test_set_x = np.load(self.cfg.test_fn_x[0])
        self.test_set_y = np.load(self.cfg.test_fn_y[0])
        self.compute_p_y_given_x(flag=3)
        return np.max(self.p_y_given_x), np.sort(np.unique(self.p_y_given_x[:, 0]), axis=0, kind='quicksort')
        # return np.min(self.p_y_given_x), np.max(self.p_y_given_x)

    def predict(self, threshold):
        self.compute_p_y_given_x(flag=3)
        return np.vstack((self.test_set_y, self.p_y_given_x[:, 0] < threshold))

    def load_data(self):
        print('... load data')
        self.train_set_x = np.load(self.cfg.train_fn_x[0])
        self.train_set_y = np.load(self.cfg.train_fn_y[0])
        self.valid_set_x = np.load(self.cfg.valid_fn_x[0])
        self.valid_set_y = np.load(self.cfg.valid_fn_y[0])
        self.test_set_x = np.load(self.cfg.test_fn_x[0])
        self.test_set_y = np.load(self.cfg.test_fn_y[0])

    def load_model(self):
        print("... load model")
        best_params = pickle.load(open(self.cfg.train_fn_model, 'rb'))
        self.W = best_params[0]
        self.b = best_params[1]

    def line_search_test(self, flag=0):
        if flag == 0:
            print('without line search ...')
        else:
            print('with line search ...')
        self.is_line_search = flag
        n_train_batches = self.train_set_x.shape[0] // self.batch_size
        validation_frequency = min(n_train_batches, self.patience // 2)
        best_validation_loss = np.inf
        start_time = timeit.default_timer()
        epoch = 0
        best_param = [self.W, self.b]

        print('trianing model ...')
        while epoch < self.num_test_epoch:
            epoch = epoch + 1

            for minibatch_index in np.arange(n_train_batches):
                self.compute_p_y_given_x(minibatch_index)
                self.gradient_w_b(minibatch_index)
                self.update_w_b(minibatch_index)

                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    this_validation_loss = self.zero_one_errors(flag=2)
                    self.test_result[epoch, flag * 2] = this_validation_loss
                    this_test_loss = self.zero_one_errors(flag=3)
                    self.test_result[epoch, flag * 2 + 1] = this_test_loss
                    # train_loss = self.negative_log_likelihood()
                    print('epoch %i, minibatch %i, letf patience %d' % (
                            epoch,
                            minibatch_index + 1,
                            self.patience - iter
                            # train_loss
                            ))
                    print('validation error %f %%, test error %f %%' % (
                            this_validation_loss * 100,
                            this_test_loss * 100
                            ))
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * self.improvement_threshold:
                            self.patience = max(self.patience, iter * self.patience_increase)
                        best_validation_loss = this_validation_loss
                        best_param = [self.W, self.b]
                if best_validation_loss < 1e-5:
                    break
        end_time = timeit.default_timer()
        print("Optimization complete with best validation loss of %f %%" % (best_validation_loss * 100))
        print("The code run for %d epochs, with %f epoch/sec, total time %.1f sec" %
              (epoch, 1.0 * epoch / (end_time - start_time), (end_time - start_time)))
        print(self.test_result)

    def draw_search_test(self, flag=0):
        if flag == 1:
            self.test_result = np.load('test_result.npy')
        epoch = np.arange(1, 201, 1)
        print(self.test_result)
        plt.plot(epoch, self.test_result[1:, 0], 'r--', epoch, self.test_result[1:, 1], 'g-')
        plt.plot(epoch, self.test_result[1:, 2], 'y--', epoch, self.test_result[1:, 3], 'b-')
        plt.show()

if __name__ == '__main__':
    cfg = config.Configuration()
    lr = LogisticRegression(cfg.lr_n_input, cfg.lr_n_class)
    # lr.load_model()
    lr.sgd_optimization()
