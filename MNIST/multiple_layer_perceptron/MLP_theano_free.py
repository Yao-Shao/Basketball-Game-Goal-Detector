import numpy as np
import os
import gzip
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
            curr_hidden_lyr = self.hidden_layer_list[i - 1]
            curr_hidden_lyr.back_compute_delta(next_W, next_delta)

            next_W = curr_hidden_lyr.W
            next_delta = curr_hidden_lyr.delta
        # then update the W and b
        xx = self.hidden_layer_list[-1].a
        self.output_layer.back_update_W_b(xx, learning_rate, L2_reg)
        for i in range(len(self.hidden_layer_list), 0, -1):
            curr_hidden_lyr = self.hidden_layer_list[i - 1]
            if i > 1:
                xx = self.hidden_layer_list[i - 2].a
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
        # self.W = np.asarray(
        #     rng.uniform(
        #         low=-np.sqrt(6. / (n_in + n_out)),
        #         high=np.sqrt(6. / (n_in + n_out)),
        #         size=(n_in, n_out)
        #     ),
        #     dtype=np.float32
        # )
        # self.W *= 4
        self.W = np.zeros(shape=(n_in, n_out), dtype=np.float32)
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
        cfg = config.Configuration()
        self.mlp = MLP(cfg.mlp_n_in, cfg.mlp_n_out)
        self.n_epochs = cfg.mlp_n_epochs
        self.patience = cfg.mlp_patience
        self.learning_rate = cfg.mlp_learning_rate
        self.batch_size = cfg.mlp_batch_size
        self.l2_reg = cfg.mlp_l2_reg
        self.improvement_threshold = cfg.mlp_improvement_threshold
        self.patience_increase = cfg.mlp_patience_increase

        # load data
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset)

        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

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
                        # best_model = copy.deepcopy(self.mlp)
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
        print('... save best model')
        with open('../model/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)


def load_data(dataset):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('loading data...')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def cvt(data_xy):
        data_x, data_y = data_xy
        data_x = np.asarray(data_x, dtype=np.float)
        data_y = np.asarray(data_y, dtype=np.int)
        return data_x, data_y

    test_set_x, test_set_y = cvt(test_set)
    valid_set_x, valid_set_y = cvt(valid_set)
    train_set_x, train_set_y = cvt(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


if __name__ == '__main__':
    opt = MlpOptimization()
    opt.mlp_optimization()