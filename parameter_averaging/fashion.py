#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import Logger
from utils import PickleUtil

class ConnectionGraph:
    edge_list = {
        "onehop": [
            (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,0),
        ],
        "twohop": [
            (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,0),
            (0,2), (1,3), (2,4), (3,5), (4,6), (5,7), (6,8), (7,9), (8,0), (9,1),
        ],
        "threehop": [
            (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,0),
            (0,2), (1,3), (2,4), (3,5), (4,6), (5,7), (6,8), (7,9), (8,0), (9,1),
            (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,0), (8,1), (9,2),
        ],
        "fc": [
            (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,0),
            (0,2), (1,3), (2,4), (3,5), (4,6), (5,7), (6,8), (7,9), (8,0), (9,1),
            (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9), (7,0), (8,1), (9,2),
            (0,4), (1,5), (2,6), (3,7), (4,8), (5,9), (6,0), (7,1), (8,2), (9,3),
            (0,5), (1,6), (2,7), (3,8), (4,9), #(5,0), (6,1), (7,2), (8,3), (9,4),
        ]
    }

class PredModel(tf.keras.Model):
    def __init__(self, name, dropout_cnn, dropout_fc):
        super(PredModel, self).__init__()
        self.model_name = name
        self.create_model(dropout_cnn, dropout_fc)
        self.build(input_shape=(None,28,28,1))

    @tf.function
    def call(self, x, training):
        # CNN 1
        h = self.conv1(x)
        #h = self.bn1(h, training=training)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.drpo1(h, training)
        # CNN2
        h = self.conv2(h)
        #h = bn2(h, training=training)
        h = self.relu2(h)
        h = self.pool2(h)
        h = self.drpo2(h, training)
        # Flatten
        h = self.fltn(h)
        # FC 1
        h = self.fc1(h)
        #h = fc_bn1(h, training=training)
        h = self.relu_fc(h)
        h = self.drpo_fc(h, training)
        # FC 2
        h = self.fc2(h)
        return h

    def create_model(self, dropout_cnn, dropout_fc):
        # CNN 1
        self.set_namescope("cnn1") # [WA:001]
        self.conv1 = tf.keras.layers.Conv2D(
            32, # filters : number of output channels
            5, # kernel size : integer or tuple
            strides=1, # integer or tuple
            padding='same', # valid or same (default valid)
            activation=None, # activation, (default None)
            name=self.layer_name('conv1')
        )
        #bn1 = tf.layers.BatchNormalization(trainable=trainable, name="bn1")
        self.relu1 = tf.keras.layers.ReLU(name=self.layer_name("relu1"))
        self.pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=2, strides=2,
            padding='same', name=self.layer_name("pool1"))
        self.drpo1 = tf.keras.layers.Dropout(rate=dropout_cnn, name=self.layer_name("dropout1"))

        # CNN 2
        self.set_namescope("cnn2") # [WA:001]
        self.conv2 = tf.keras.layers.Conv2D(
            64, # filters : number of output channels
            5, # kernel size : integer or tuple
            strides=1, # integer or tuple
            padding='same', # valid or same (default valid)
            activation=None, # activation, (default None)
            name=self.layer_name('conv2')
        )
        #bn2 = tf.layers.BatchNormalization(trainable=trainable, name="bn2")
        self.relu2 = tf.keras.layers.ReLU(name=self.layer_name("relu2"))
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=2, strides=2,
            padding='same', name=self.layer_name("pool2"))
        self.drpo2 = tf.keras.layers.Dropout(rate=dropout_cnn, name=self.layer_name("dropout2"))

        # Flatten
        self.set_namescope("fc") # [WA:001]
        self.fltn = tf.keras.layers.Flatten(name=self.layer_name("flatten"))

        # FC 1
        self.fc1 = tf.keras.layers.Dense(512, activation=None, name=self.layer_name("fc1"))
        #fc_bn1 = tf.layers.BatchNormalization(trainable=trainable, name="fc_bn1")
        self.relu_fc = tf.keras.layers.ReLU(name=self.layer_name("relu_fc"))
        self.drpo_fc = tf.keras.layers.Dropout(rate=dropout_fc, name=self.layer_name("dropout_fc"))

        # FC 2
        self.fc2 = tf.keras.layers.Dense(10, activation=None, name=self.layer_name("fc2"))

        # [WA:001]
        self.set_namescope = None
        self.scope = None
        self.layer_name = None

    # WA [WA:001] (tf.name_scope does not work with tf2.0+keras)
    def set_namescope(self, scope):
        self.scope = scope
    def layer_name(self, name):
        return "{}/{}/{}".format(self.model_name, self.scope, name)

class Agent:
    def __init__(
            self,
            dropout_cnn, dropout_fc,
            learning_rate, averaging_rate,
            minibatchsize, minibatchsize_shared,
            minibatchsize_test,
            name):
        self.name = name

        self.dropout_cnn = dropout_cnn
        self.dropout_fc = dropout_fc
        self.learning_rate = learning_rate
        self.averaging_rate = averaging_rate
        self.build_model()

        #print(name)
        #self.model.summary()

        self.adjacent = []

        self.minibatchsize = minibatchsize
        self.minibatchsize_shared = minibatchsize_shared
        self.minibatchsize_test = minibatchsize_test

    def build_model(self):
        self.model = PredModel(self.name, self.dropout_cnn, self.dropout_fc)

        # losses
        self.loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        #self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # metrics
        self.met_loss = tf.keras.metrics.Mean()
        self.met_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.met_loss_val = tf.keras.metrics.Mean()
        self.met_acc_val = tf.keras.metrics.SparseCategoricalAccuracy()

    def make_ds(self, x, y, batchsize, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x))
        ds = ds.batch(batchsize)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def set_train_data(self, train_x, train_y):
        self.train_ds = self.make_ds(
            train_x, train_y,
            self.minibatchsize,
            shuffle=True)

    def set_test_data(self, test_x, test_y):
        self.test_ds = self.make_ds(
            test_x, test_y,
            self.minibatchsize_test,
            shuffle=False
        )

    def train(self):
        self.met_loss.reset_states()
        self.met_acc.reset_states()
        for x,y in self.train_ds:
            self.train_step(x, y)
        return (self.met_loss.result().numpy(),
                self.met_acc.result().numpy())

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            pred_y = self.model(x, training=True)
            loss = self.loss_op(y, pred_y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.met_loss.update_state(loss)
        self.met_acc.update_state(y, pred_y)

    def copy_weight(self):
        self.dummy_weight = [v.read_value() for v in self.model.trainable_variables]

    def averaging_weight(self):
        adj_vars = (ag.dummy_weight for ag in self.adjacent)
        for dist, src in zip(self.model.trainable_variables,
                             (tf.reduce_mean(v, axis=0) for v in zip(*adj_vars))):
            dist.assign(self.averaging_rate*dist + (1-self.averaging_rate)*src)

    def validation(self):
        self.met_loss_val.reset_states()
        self.met_acc_val.reset_states()
        for x,y in self.test_ds:
            self.valid_step(x, y)
        return (self.met_loss_val.result().numpy(),
                self.met_acc_val.result().numpy())

    @tf.function
    def valid_step(self, x, y):
        pred_y = self.model(x, training=False)
        loss = self.loss_op(y, pred_y)

        self.met_loss_val.update_state(loss)
        self.met_acc_val.update_state(y, pred_y)

class Simulator:
    def __init__(self, prm):
        self.set_params(prm)
        self.load_dataset()
        self.set_agents()

    def set_params(self, prm):
        # default parameters
        # mnist
        self.num_label = 10
        self.size = 28
        self.max_input_val = 255.0
        self.num_train_data_all = 60000
        self.num_test_data_all = 10000
        # simulation
        # agent settings
        self.num_ags = 10
        self.topology = "onehop"
        # data settings
        self.num_train_data = 10000
        # num of train data per agent : num_train_data / num_ags
        self.num_labels_ag = 2 # 0: IID
        # learning settings
        self.minibatchsize = 100
        self.minibatchsize_test = 500
        self.num_batch = self.num_train_data / self.minibatchsize
        self.minibatchsize_shared = 100
        self.initialize_weight_same = True
        self.learning_rate = 0.01
        self.averaging_rate = 0.9 # sharing rate
        # self * avearging_rate + (1-averaging_rate) * (average of adjacent)
        self.dropout_fc = 0.1
        self.dropout_cnn = 0.5
        self.num_epoch = 1000
        self.show_step_num = 10
        self.dtype = "float32"

        for k, v in prm.items():
            setattr(self, k, v)

        self.params = self.__dict__.copy()

    def load_fashion(self):
        self.train_data, self.test_data = tf.keras.datasets.fashion_mnist.load_data()
        self.train_x, self.train_y = self.train_data
        self.test_x, self.test_y = self.test_data
        self.train_x = self.train_x / self.max_input_val - 0.5
        self.test_x = self.test_x / self.max_input_val - 0.5
        self.train_x = self.train_x[..., tf.newaxis]
        self.test_x = self.test_x[..., tf.newaxis]
        if self.dtype == "float32":
            npdtype = np.float32
        self.train_x = np.array(self.train_x, dtype=npdtype)
        self.test_x = np.array(self.test_x, dtype=npdtype)

    def load_dataset(self):
        self.load_fashion()
        self.decide_data_per_agents()
        #self.set_shared_data()

    def decide_data_per_agents(self):
        num_data_per_ag = self.num_train_data // self.num_ags

        if self.num_labels_ag == 0:
            perm = np.random.permutation(self.num_train_data)
            self.data_slice = perm.reshape(self.num_ags, num_data_per_ag)
        else:
            def random_sample(label):
                indices = np.where(self.train_y==label)[0]
                return indices[np.random.permutation(len(indices))][0:num_data_per_ag]
            label2idx = [random_sample(i) for i in range(self.num_label)]

            ndl = num_data_per_ag // self.num_labels_ag
            self.data_slice = np.array([
                np.concatenate([
                    label2idx[(i+k)%self.num_ags][ndl*k:ndl*(k+1)]
                    for k in range(self.num_labels_ag)
                ])
                for i in range(self.num_ags)
            ])
        assert(self.data_slice.shape == (self.num_ags, num_data_per_ag))

    def set_agents(self):
        self.ags = [Agent(
                        self.dropout_cnn, self.dropout_fc,
                        self.learning_rate, self.averaging_rate,
                        self.minibatchsize, self.minibatchsize_shared,
                        self.minibatchsize_test,
                        "agent{0}".format(i))
                    for i in range(self.num_ags)]
        # set data
        for i, ag in enumerate(self.ags):
            ag.set_train_data(self.train_x[self.data_slice[i]],
                             self.train_y[self.data_slice[i]])
            ag.set_test_data(self.test_x, self.test_y)

        # set connection
        for i,j in ConnectionGraph.edge_list[self.topology]:
            self.ags[i].adjacent.append(self.ags[j])
            self.ags[j].adjacent.append(self.ags[i])

    def train(self):
        train_l = np.zeros(self.num_ags)
        train_a = np.zeros(self.num_ags)
        for i, ag in enumerate(self.ags):
            train_l[i], train_a[i] = ag.train()
        return train_l, train_a

    def averaging_weight(self):
        for ag in self.ags:
            ag.copy_weight()
        for ag in self.ags:
            ag.averaging_weight()

    def validation(self):
        val_l = np.zeros(self.num_ags)
        val_a = np.zeros(self.num_ags)
        for i, ag in enumerate(self.ags):
            val_l[i], val_a[i] = ag.validation()
        return val_l, val_a

    def run(self):
        results = {}
        results["params"] = self.params
        log = Logger()
        start_time = time.time()
        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []
        clock_time = []

        if self.initialize_weight_same:
            for i in range(1, self.num_ags):
                for dist, src in zip(self.ags[i].model.trainable_variables,
                                     self.ags[0].model.trainable_variables):
                    dist.assign(src)
        try:
            for epoch in range(self.num_epoch):
                train_l, train_a = self.train()
                self.averaging_weight()
                val_l, val_a = self.validation()

                train_loss.append(train_l)
                train_acc.append(train_a)
                valid_loss.append(val_l)
                valid_acc.append(val_a)
                clock_time.append(time.time() - start_time)

                if (epoch + 1) % self.show_step_num == 0:
                    log("epoch {:>3}".format(epoch))
                    log("  loss(train): " + Logger.strarray(train_l))
                    log("  acc.(train): " + Logger.strarray(train_a))
                    log("  loss(valid): " + Logger.strarray(val_l))
                    log("  acc.(valid): " + Logger.strarray(val_a))
                # tmp data
                results["data"] = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_acc,
                    "time" : clock_time
                }
                results["logs"] = log
                PickleUtil.save_data(results, "debug/running")
        except KeyboardInterrupt:
            log("stopped {:>3}".format(epoch))

        results["data"] = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_acc,
            "time" : clock_time
        }
        results["logs"] = log
        self.results = results
        return results

def set_tf_config(dtype):
    tf.keras.backend.set_floatx(dtype)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Physical GPUs: {}, Logical GPUs: {}".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if __name__ == "__main__":
    raise Exception("Never run this file directly")

