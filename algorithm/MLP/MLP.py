import numpy as np
import timeit
import pickle
import copy
from preprocess import config


class MLP:
    def __init__(self, n_in=28*28, n_out=10,
                 n_list_hidden_nodes=[500]):
        rng = np.random.RandomState(1234)
        # init the hidden layers
        self.hidden_layer_list = []
        last_n_nodes = n_in
        for i in n_list_hidden_nodes:
            self.hidden_layer_list.append(HiddenLayer(last_n_nodes, i, rng))
            last_n_nodes = i
        # init the output layer
        self.output_layer = OutputLayer(last_n_nodes, n_out, rng)

    def feedforward(self, x):
        xx = x
        for item in self.hidden_layer_list:
            item.forward_compute_z_a(xx)
            xx = item.a

        self.output_layer.forward_compute_p_y_given_x(xx)

    def backpropagation(self, x, y, learning_rate, L2_reg):
        # first compute all the delta in every layer
        self.output_layer.back_compute_delta(y)
        next_delta = self.output_layer.delta
        next_W = self.output_layer.W
        for i in range(len(self.hidden_layer_list), 0, -1):
            curr_hidden_lyr = self.hidden_layer_list[i-1]
            curr_hidden_lyr.back_compute_delta(next_W, next_delta)

            next_W = curr_hidden_lyr.W
            next_delta = curr_hidden_lyr.delta
        # then update the W and b
        xx = self.hidden_layer_list[-1].a
        self.output_layer.back_update_W_b(xx, learning_rate, L2_reg)
        for i in range(len(self.hidden_layer_list), 0, -1):
            curr_hidden_lyr = self.hidden_layer_list[i - 1]
            if i > 1:
                xx = self.hidden_layer_list[i-2].a
            else:
                xx = x
            curr_hidden_lyr.back_update_W_b(xx, learning_rate, L2_reg)

class HiddenLayer():
    def __init__(self, n_in, n_out, rng):
        self.W = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=np.float32
        )
        self.b = np.zeros(shape=(n_out,), dtype=np.float32)
        self.a = None
        self.z = None
        self.delta = None

    def forward_compute_z_a(self, x):
        self.z = np.dot(x, self.W) + self.b
        self.a = np.tanh(self.z)
        return self.a

    def back_compute_delta(self, next_W, next_delta):
        tt = np.dot(next_delta, next_W.transpose())
        self.delta = tt * (1 - self.a ** 2) # f'(z)

    def back_update_W_b(self, x, learning_rate, L2_reg):
        delta_W = -1.0 * np.dot(x.transpose(), self.delta) / x.shape[0]
        delta_b = -1.0 * np.mean(self.delta, axis=0)

        self.W -= learning_rate * (L2_reg * self.W + delta_W)
        self.b -= learning_rate * delta_b


class OutputLayer():
    def __init__(self, n_in, n_out, rng):
        self.n_out = n_out
        self.W = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=np.float32
        )
        self.W *= 4
        self.b = np.zeros(shape=(n_out,), dtype=np.float32)
        self.p_y_given_x = None
        self.delta = None

    def forward_compute_p_y_given_x(self, x):
        self.exp_x_multiply_W_plus_b = np.exp(np.dot(x, self.W) + self.b)
        sigma = np.sum(self.exp_x_multiply_W_plus_b, axis=1)
        self.p_y_given_x = self.exp_x_multiply_W_plus_b / sigma.reshape(sigma.shape[0], 1) # transpose

    def back_compute_delta(self, y):
        yy = np.zeros((y.shape[0], self.n_out))
        yy[np.arange(y.shape[0]), y] = 1.0
        self.delta = yy - self.p_y_given_x

    def back_update_W_b(self, x, learning_rate, L2_reg):
        delta_W = -1.0 * np.dot(x.transpose(), self.delta) / x.shape[0]
        delta_b = -1.0 * np.mean(self.delta, axis=0)
        self.W -= learning_rate * (delta_W + L2_reg * self.W)
        self.b -= learning_rate * delta_b


class MlpOptimization():
    def __init__(self):
        cfg = Configuration.Configuration()
        self.n_in = cfg.mlp_n_in
        self.n_out = cfg.mlp_n_out
        self.n_list_hidden_nodes = cfg.mlp_n_list_hidden_nodes
        self.n_epochs = cfg.mlp_n_epochs
        self.patience = cfg.mlp_patience
        self.learning_rate = cfg.mlp_learning_rate
        self.batch_size = cfg.mlp_batch_size
        self.l2_reg = cfg.mlp_l2_reg
        self.improvement_threshold = cfg.mlp_improvement_threshold
        self.patience_increase = cfg.mlp_patience_increase

        self.mlp = MLP(self.n_in, self.n_out, self.n_list_hidden_nodes)

        # load data
        # print("loading data...")
        video_index = 2
        self.train_set_x = np.load(cfg.train_x_fn[video_index])
        self.train_set_y = np.load(cfg.train_y_fn[video_index])
        self.valid_set_x = np.load(cfg.valid_x_fn[video_index])
        self.valid_set_y = np.load(cfg.valid_y_fn[video_index])
        self.test_set_x = np.load(cfg.test_x_fn[video_index])
        self.test_set_y = np.load(cfg.test_y_fn[video_index])

    def zero_one_errors(self, index=0, flag=1):
        if flag == 1:
            x = self.train_set_x[index * self.batch_size
                                 : (index + 1) * self.batch_size]
            y = self.train_set_y[index * self.batch_size
                                 : (index + 1) * self.batch_size]
        elif flag == 2:
            # tt = int(self.valid_set_y.shape[0] / self.batch_size)
            # y = self.valid_set_y[0: tt * self.batch_size]
            x = self.valid_set_x
            y = self.valid_set_y
        else:
            # tt = int(self.test_set_y.shape[0] / self.batch_size)
            # y = self.test_set_y[0: tt * self.batch_size]
            x = self.test_set_x
            y = self.test_set_y

        self.mlp.feedforward(x)
        predict_y = np.argmax(self.mlp.output_layer.p_y_given_x, axis=1)
        return np.mean(predict_y != y)

    def mlp_optimization(self):

        # compute number of minibatches for training, validation and testing
        n_train_batches = self.train_set_x.shape[0] // self.batch_size

        validation_frequency = min(n_train_batches, self.patience // 2)

        best_validation_loss = np.inf
        test_error = np.inf

        epoch = 0
        done_looping = False

        best_model = copy.deepcopy(self.mlp)

        print('trianing model...')

        start_time = timeit.default_timer()
        while (epoch < self.n_epochs) and (not done_looping):
            epoch_start_time = timeit.default_timer()
            epoch = epoch + 1
            for minibatch_index in np.arange(n_train_batches):
                x = self.train_set_x[minibatch_index * self.batch_size
                                     : (minibatch_index + 1) * self.batch_size]
                y = self.train_set_y[minibatch_index * self.batch_size
                                     : (minibatch_index + 1) * self.batch_size]
                self.mlp.feedforward(x)
                self.mlp.backpropagation(x, y, self.learning_rate, self.l2_reg)

                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    this_validation_loss = self.zero_one_errors(flag=2)
                    epoch_end_time = timeit.default_timer()
                    print('epoch %i, minibatch %i/%i, left patience %d, validation error %f %%, time %.2fs' % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        self.patience - iter,
                        this_validation_loss * 100,
                        epoch_end_time - epoch_start_time))
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * self.improvement_threshold:
                            self.patience = max(self.patience, iter * self.patience_increase)
                        best_validation_loss = this_validation_loss
                        best_model = copy.deepcopy(self.mlp)
                        test_error = self.zero_one_errors(flag=3) * 100
                        print('\tepoch %i, minibatch %i/%i, test error of best model %f %%' % (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_error))
                if self.patience <= iter:
                    done_looping = True
                    break
                if best_validation_loss < 1e-5:
                    break
        end_time = timeit.default_timer()
        print("Optimization complete with best validation loss of %f %%, test loss of %f %%" %
              (best_validation_loss * 100, test_error))
        print("The code run for %d epochs, with %f epoch/sec, total time %.1f sec" %
              (epoch, 1.0 * epoch / (end_time - start_time), (end_time - start_time)))
        # save best model
        print('save best model...')
        with open('../model/' + 'hy_' + str(len(self.n_list_hidden_nodes))
                  + '_hn_' + str(self.n_list_hidden_nodes[0]) + '.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print('save probility...')
        best_model.feedforward(self.test_set_x)
        np.save('../dataset/probility/MLP_lr_' + str(self.learning_rate) + '_hd_' + str(len(self.n_list_hidden_nodes))
                + '_hn_' + str(self.n_list_hidden_nodes) + '_Fea_2', self.mlp.output_layer.p_y_given_x)

    def get_probility(self):
        # with open('../dateset/probility/MLP_hd_' + str(len(self.n_list_hidden_nodes))
        #           + '_hn_' + str(self.n_list_hidden_nodes[0]) + '_Fea_2_50*50.pkl', 'wb') as f:
        #     pickle.dump(self.mlp.output_layer.p_y_given_x, f)
        self.mlp = pickle.load(open('../model/hy_1_hn_100.pkl', 'rb'))
        # self.mlp = pickle.load(open('../model/' + 'hy_' + str(len(self.n_list_hidden_nodes))
        #           + '_hn_' + str(self.n_list_hidden_nodes[0]) + '.pkl', 'rb'))
        self.mlp.feedforward(self.test_set_x)
        np.save('../dataset/probility/MLP_lr_' + str(self.learning_rate) + '_hd_' + str(len(self.n_list_hidden_nodes))
                  + '_hn_' + str(self.n_list_hidden_nodes) + '_Fea_2', self.mlp.output_layer.p_y_given_x)

    def predict(self, threshold, prob_fn):
        # probility = pickle.load(open(prob_fn, 'rb'))
        probility = np.load(prob_fn)
        return probility[:, 1] >= threshold, self.test_set_y

    def get_threshold_range(self, prob_fn):
        # probility = pickle.load(open(prob_fn, 'rb'))
        probility = np.load(prob_fn)
        return np.sort(np.unique(probility[:, 1]), axis=0, kind='quicksort')

if __name__ == '__main__':
    opt = MlpOptimization()
    opt.mlp_optimization()