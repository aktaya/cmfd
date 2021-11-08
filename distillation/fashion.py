#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
import numpy as np
import networkx as nx
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
        ],
        "fixed_ba": [
            (0, 1), (0, 2), (1, 4), (1, 5), (1, 9), (2, 3), (2, 6), (2, 8), (6, 7)
        ],
        "fixed_ba2": [
            (0, 3), (0, 4), (0, 5), (0, 7), (0, 9),
            (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 3), (2, 8),
            (3, 4), (3, 6), (3, 7), (3, 9),
            (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
            (7, 8)
        ],
        "fixed_ba100": [
            (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 12), (0, 13), (0, 14), (0, 15), (0, 17),
            (0, 22), (0, 28), (0, 29), (0, 32), (0, 38), (0, 40), (0, 46), (0, 68), (0, 78), (0, 82), (0, 83), (0, 98),
            (1, 3), (1, 4), (1, 5), (1, 8), (1, 26), (1, 50), (1, 77), (1, 88), (2, 3), (2, 7), (2, 8), (2, 13),
            (2, 28), (2, 44), (2, 51), (2, 78), (3, 4), (3, 6), (3, 7), (3, 9), (3, 10), (3, 11), (3, 14), (3, 15),
            (3, 16), (3, 19), (3, 26), (3, 31), (3, 32), (3, 34), (3, 35), (3, 41), (3, 45), (3, 46), (3, 47), (3, 54),
            (3, 59), (3, 63), (3, 67), (3, 70), (3, 87), (3, 90), (3, 95), (3, 98), (4, 5), (4, 11), (4, 16), (4, 17),
            (4, 20), (4, 22), (4, 24), (4, 30), (4, 33), (4, 57), (4, 59), (4, 61), (4, 68), (4, 69), (4, 75), (4, 76),
            (4, 82), (4, 96), (5, 6), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 18), (5, 19), (5, 23), (5, 27),
            (5, 29), (5, 31), (5, 45), (5, 47), (5, 48), (5, 50), (5, 56), (5, 62), (5, 78), (5, 80), (5, 85), (5, 91), (5, 92),
            (6, 10), (6, 12), (6, 42), (6, 58), (6, 89), (7, 26), (7, 38), (7, 52), (7, 57), (7, 66), (7, 67), (7, 73), (7, 84),
            (8, 14), (8, 15), (8, 16), (8, 21), (8, 22), (8, 25), (8, 27), (8, 28), (8, 30), (8, 35), (8, 39), (8, 43), (8, 44),
            (8, 51), (8, 55), (8, 62), (8, 68), (8, 72), (8, 81), (8, 89), (8, 93), (8, 94), (9, 20), (9, 40), (9, 48), (10, 25),
            (10, 35), (10, 95), (10, 99), (11, 18), (11, 19), (11, 21), (11, 23), (11, 34), (11, 36), (11, 49), (11, 58),
            (11, 76), (11, 89), (12, 34), (12, 42), (12, 50), (12, 62), (12, 63), (12, 92), (13, 20), (13, 24), (14, 17),
            (14, 18), (14, 33), (14, 44), (14, 61), (14, 84), (15, 24), (16, 55), (16, 57), (16, 72), (16, 75), (17, 49), (17, 79),
            (18, 29), (18, 98), (19, 37), (19, 47), (19, 71), (20, 21), (20, 39), (20, 72), (20, 85), (20, 93), (21, 41),
            (22, 23), (22, 36), (22, 56), (22, 73), (22, 74), (23, 25), (23, 31), (23, 40), (23, 99), (24, 55), (24, 63),
            (24, 64), (24, 79), (24, 83), (24, 88), (24, 90), (25, 59), (25, 84), (26, 27), (26, 58), (27, 36), (27, 37),
            (27, 38), (27, 52), (27, 70), (27, 77), (27, 96), (27, 97), (28, 30), (29, 37), (29, 54), (30, 42), (30, 77), (30, 88),
            (31, 32), (31, 33), (31, 41), (31, 48), (31, 53), (31, 94), (31, 95), (32, 66), (32, 87), (34, 86), (34, 96),
            (35, 53), (35, 64), (35, 80), (36, 39), (36, 43), (36, 49), (36, 61), (36, 67), (36, 79), (36, 83), (38, 74),
            (38, 85), (38, 93), (39, 43), (40, 54), (41, 45), (41, 56), (41, 60), (41, 71), (41, 74), (42, 46), (43, 66),
            (43, 71), (43, 81), (44, 60), (44, 69), (44, 81), (45, 51), (45, 65), (45, 73), (46, 53), (46, 82), (46, 99),
            (48, 75), (49, 52), (51, 60), (54, 80), (55, 64), (56, 65), (60, 76), (63, 65), (63, 69), (63, 91), (65, 87),
            (67, 70), (70, 90), (70, 97), (75, 86), (77, 94), (80, 86), (83, 91), (84, 97), (91, 92)
        ],
    }
    fixed_topologies = edge_list.keys()

    def ba_model_edge_list(num_nodes, m):
        g = nx.barabasi_albert_graph(num_nodes, m)
        return g.edges

class PredModel0(tf.keras.Model):
    def __init__(self, name, dropout_cnn, dropout_fc):
        super(PredModel0, self).__init__()
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

class PredModel1(tf.keras.Model):
    def __init__(self, name, dropout_cnn, dropout_fc):
        super(PredModel1, self).__init__()
        self.model_name = name
        self.create_model(dropout_cnn, dropout_fc)
        self.build(input_shape=(None,28,28,1))

    @tf.function
    def call(self, x, training):
        # CNN 1
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.drpo1(h, training)
        # CNN2
        h = self.conv2(h)
        #h = bn2(h, training=training)
        h = self.relu2(h)
        h = self.pool2(h)
        h = self.drpo2(h, training)
        # CNN3
        h = self.conv3(h)
        h = self.relu3(h)
        h = self.pool3(h)
        h = self.drpo3(h, training)
        # Flatten
        h = self.fltn(h)
        # FC 1
        h = self.fc1(h)
        h = self.relu_fc1(h)
        h = self.drpo_fc1(h, training)
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
            3, # kernel size : integer or tuple
            strides=1, # integer or tuple
            padding='same', # valid or same (default valid)
            activation=None, # activation, (default None)
            name=self.layer_name('conv2')
        )
        #bn2 = tf.layers.BatchNormalization(trainable=trainable, name="bn2")
        self.relu2 = tf.keras.layers.ReLU(name=self.layer_name("relu2"))
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=1, strides=1,
            padding='same', name=self.layer_name("pool2"))
        self.drpo2 = tf.keras.layers.Dropout(rate=dropout_cnn, name=self.layer_name("dropout2"))

        # CNN 3
        self.set_namescope("cnn3") # [WA:001]
        self.conv3 = tf.keras.layers.Conv2D(
            64, # filters : number of output channels
            3, # kernel size : integer or tuple
            strides=1, # integer or tuple
            padding='same', # valid or same (default valid)
            activation=None, # activation, (default None)
            name=self.layer_name('conv3')
        )
        self.relu3 = tf.keras.layers.ReLU(name=self.layer_name("relu3"))
        self.pool3 = tf.keras.layers.MaxPooling2D(
            pool_size=2, strides=2,
            padding='same', name=self.layer_name("pool3"))
        self.drpo3 = tf.keras.layers.Dropout(rate=dropout_cnn, name=self.layer_name("dropout3"))

        # Flatten
        self.set_namescope("fc") # [WA:001]
        self.fltn = tf.keras.layers.Flatten(name=self.layer_name("flatten"))

        # FC 1
        self.fc1 = tf.keras.layers.Dense(512, activation=None, name=self.layer_name("fc1"))
        self.relu_fc1 = tf.keras.layers.ReLU(name=self.layer_name("relu_fc1"))
        self.drpo_fc1 = tf.keras.layers.Dropout(rate=dropout_fc, name=self.layer_name("dropout_fc1"))

        # FC 2
        self.fc2 = tf.keras.layers.Dense(10, activation=None, name=self.layer_name("fc2"))
        self.set_namescope = None
        self.scope = None
        self.layer_name = None

    # WA [WA:001] (tf.name_scope does not work with tf2.0+keras)
    def set_namescope(self, scope):
        self.scope = scope
    def layer_name(self, name):
        return "{}/{}/{}".format(self.model_name, self.scope, name)

class PredModelMini(tf.keras.Model):
    def __init__(self, name, dropout_cnn, dropout_fc):
        super(PredModelMini, self).__init__()
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
            8, # filters : number of output channels
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

        # Flatten
        self.set_namescope("fc") # [WA:001]
        self.fltn = tf.keras.layers.Flatten(name=self.layer_name("flatten"))

        # FC 1
        self.fc1 = tf.keras.layers.Dense(32, activation=None, name=self.layer_name("fc1"))
        #fc_bn1 = tf.layers.BatchNormalization(trainable=trainable, name="fc_bn1")
        self.relu_fc = tf.keras.layers.ReLU(name=self.layer_name("relu_fc"))
        self.drpo_fc = tf.keras.layers.Dropout(rate=dropout_fc, name=self.layer_name("dropout_fc"))

        # FC 2
        self.fc2 = tf.keras.layers.Dense(10, activation=None, name=self.layer_name("fc2"))

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
            model_type,
            mean_after_softmax,
            max_averaging,
            dropout_cnn, dropout_fc,
            codis_loss,
            optimizer, learning_rate, decay_rate,
            learning_rate_codis,
            use_momentum, learning_rate_codis_mom, momentum,
            minibatchsize, minibatchsize_shared,
            minibatchsize_test,
            name):
        self.name = name

        self.model_type = model_type
        self.mean_after_softmax = mean_after_softmax
        self.max_averaging = max_averaging
        self.dropout_cnn = dropout_cnn
        self.dropout_fc = dropout_fc
        self.codis_loss = codis_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.learning_rate_c = learning_rate_codis
        self.learning_rate_cmom = learning_rate_codis_mom * (1-momentum)
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.build_model()
        self.codis_y = tf.Variable(
            initial_value=0.0,
            validate_shape=False,
            trainable=False,
            shape=tf.TensorShape(None))

        #print(name)
        #self.model.summary()

        self.adjacent = []

        self.minibatchsize = minibatchsize
        self.minibatchsize_shared = minibatchsize_shared
        self.minibatchsize_test = minibatchsize_test

    def build_model(self):
        if self.model_type == 0:
            self.model = PredModel0(self.name, self.dropout_cnn, self.dropout_fc)
        elif self.model_type == 1:
            self.model = PredModel1(self.name, self.dropout_cnn, self.dropout_fc)
        elif self.model_type == 2:
            self.model = PredModelMini(self.name, self.dropout_cnn, self.dropout_fc)
        else:
            raise NotImplementedError

        # losses
        self.loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if self.codis_loss == "mse":
            self.loss_op_codis = tf.keras.losses.MeanSquaredError()
        elif self.codis_loss == "cross_entropy":
            self.loss_op_codis = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        else:
            raise NotImplementedError

        if self.optimizer == "sgd":
            self.opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == "itd":
            self.opt = tf.keras.optimizers.SGD(learning_rate=InverseTimeDecay(
                self.learning_rate, 1, self.decay_rate))
        elif self.optimizer == "sitd":
            self.opt = tf.keras.optimizers.SGD(learning_rate=SquareInverseTimeDecay(
                self.learning_rate, 1, self.decay_rate))
        elif self.optimizer == "adam":
            self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            raise NotImplementedError

        self.opt_codis = tf.keras.optimizers.SGD(learning_rate=self.learning_rate_c)
        if self.use_momentum:
            self.opt_codis_mom = tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate_cmom, momentum=self.momentum)

        # metrics
        self.met_loss = tf.keras.metrics.Mean()
        self.met_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.met_loss_codis = tf.keras.metrics.Mean()
        #self.met_acc_codis = tf.keras.metrics.CategoricalAccuracy()
        self.met_loss_val = tf.keras.metrics.Mean()
        self.met_acc_val = tf.keras.metrics.SparseCategoricalAccuracy()

    def adjust_lr_codis(self):
        lrc = self.opt_codis.learning_rate
        self.opt_codis.learning_rate = lrc * len(self.adjacent)

    @tf.function
    def adjacent_mean(self):
        return tf.reduce_mean([ag.codis_y for ag in self.adjacent], axis=0)

    @tf.function
    def adjacent_max(self):
        adj_codis_y = tf.stack([ag.codis_y for ag in self.adjacent])
        max_idx = tf.reduce_max(adj_codis_y, axis=2)
        max_adj = tf.cast(tf.argmax(max_idx, axis=0), dtype=tf.int32)
        indices = tf.concat([max_adj[:,tf.newaxis], tf.range(len(max_adj))[:,tf.newaxis]], axis=1)
        return tf.gather_nd(adj_codis_y, indices)

    @tf.function
    def _calc_codistillation_data(self, x):
        if self.mean_after_softmax:
            return tf.nn.softmax(self.model(x, training=False))
        else:
            return self.model(x, training=False)

    @tf.function
    def calc_codis_data(self, shared_data):
        self.codis_y.assign(self._calc_codistillation_data(shared_data))

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

    def set_codis_data(self, codis_x):
        if self.max_averaging:
            reduced_codis_y = self.adjacent_max()
        else:
            reduced_codis_y = self.adjacent_mean()
        self.codis_ds = self.make_ds(
            codis_x, reduced_codis_y,
            self.minibatchsize_shared,
            shuffle=True
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

    def codistillation(self):
        self.met_loss_codis.reset_states()
        #self.met_acc_codis.reset_states()
        for x,y in self.codis_ds:
            self.codis_step(x, y)
        #return (self.met_loss_codis.result().numpy(),
        #        self.met_acc_codis.result().numpy())
        return self.met_loss_codis.result().numpy()

    @tf.function
    def codis_step(self, x, y):
        with tf.GradientTape() as tape:
            if self.mean_after_softmax:
                pred_y = tf.nn.softmax(self.model(x, training=True))
            else:
                pred_y = self.model(x, training=True)
            loss = self.loss_op_codis(y, pred_y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt_codis.apply_gradients(zip(gradients, self.model.trainable_variables))
        if self.use_momentum:
            self.opt_codis_mom.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.met_loss_codis.update_state(loss)
        #self.met_acc_codis.update_state(y, pred_y)

    def train_codis(self):
        self.met_loss.reset_states()
        self.met_acc.reset_states()
        self.met_loss_codis.reset_states()
        for (tx,ty), (cx,cy) in zip(self.train_ds, self.codis_ds):
            self.train_step(tx, ty)
            self.codis_step(cx, cy)
        return (self.met_loss.result().numpy(),
                self.met_acc.result().numpy(),
                self.met_loss_codis.result().numpy())

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
        self.start_m = None # only used when topology == "ba_model"
        # data settings
        self.num_train_data = 10000
        # num. of train data per agent : num_train_data / num_ags
        self.num_labels_ag = 2 # 0: IID
        self.num_shared = 10000
        # learning settings
        self.minibatchsize = 100
        self.minibatchsize_test = 500
        #self.num_batch = self.num_train_data / self.minibatchsize
        self.minibatchsize_shared = 100
        self.codis_loss = "mse" # mse or cross_entropy
        self.optimizer = "sgd"
        self.initialize_weight_same = False
        self.learning_rate = 0.01
        self.decay_rate = 0
        self.learning_rate_codis = 1 # sharing rate
        self.adjust_lr_codis = False
        self.max_averaging = False
        self.use_momentum = False
        self.learning_rate_codis_mom = 1
        self.momentum = 0.0
        self.nn_model = "model_0"
        self.dropout_fc = 0.1
        self.dropout_cnn = 0.5
        self.mean_after_softmax = True
        self.codis_after_train = True
        self.codis_iter = 1 # num of distillation for each epoch (only valid when codis_after_train=True)
        self.num_epoch = 1000
        self.show_step_num = 10
        self.dtype = "float32"

        for k, v in prm.items():
            setattr(self, k, v)
        # for backward compatibility
        if self.nn_model in ["model_0", "model_1", "model_mini"]:
            self.num_model_type = 1
        elif self.nn_model in ["model_0_1", "model_0_mini"]:
            self.num_model_type = 2
        else:
            raise NotImplementedError

        self.params = self.__dict__.copy()

    def load_fashionmnist(self):
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
        self.load_fashionmnist()
        self.decide_data_per_agents()
        self.set_shared_data()

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
                    label2idx[(i+k)%self.num_label][ndl*k:ndl*(k+1)]
                    for k in range(self.num_labels_ag)
                ])
                for i in range(self.num_ags)
            ])
        assert(self.data_slice.shape == (self.num_ags, num_data_per_ag))

    def set_shared_data(self):
        shared_data_slice = np.random.permutation(self.num_train_data_all)[:self.num_shared]
        self.shared_data = self.train_x[shared_data_slice]

    def set_agents(self):
        try:
            get_modeltype = {
                "model_0": (lambda i: 0),
                "model_1": (lambda i: 1),
                "model_mini": (lambda i: 2),
                "model_0_1": (lambda i: (i % 2)),
                "model_0_mini": (lambda i: 0 if (i % 2)==0 else 2),
            }[self.nn_model]
        except:
            raise NotImplementedError
        self.ags = [Agent(
                        get_modeltype(i),
                        self.mean_after_softmax,
                        self.max_averaging,
                        self.dropout_cnn, self.dropout_fc,
                        self.codis_loss,
                        self.optimizer, self.learning_rate, self.decay_rate,
                        self.learning_rate_codis,
                        self.use_momentum, self.learning_rate_codis_mom, self.momentum,
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
        self.set_connection()

        # change learning rate of codistillation
        if self.adjust_lr_codis:
            for ag in self.ags:
                ag.adjust_lr_codis()

    def set_connection(self):
        for ag in self.ags:
            ag.adjacent = []

        if self.topology in ConnectionGraph.fixed_topologies:
            edge_list = ConnectionGraph.edge_list[self.topology]
        elif self.topology == "ba_model":
            edge_list = ConnectionGraph.ba_model_edge_list(self.num_ags, self.start_m)
        else:
            raise NotImplementedError
        for i,j in edge_list:
            self.ags[i].adjacent.append(self.ags[j])
            self.ags[j].adjacent.append(self.ags[i])

    def train(self):
        train_l = np.zeros(self.num_ags)
        train_a = np.zeros(self.num_ags)
        for i, ag in enumerate(self.ags):
            train_l[i], train_a[i] = ag.train()
        return train_l, train_a

    def codistillation(self):
        for ag in self.ags:
            ag.calc_codis_data(self.shared_data)

        for ag in self.ags:
            ag.set_codis_data(self.shared_data)

        codis_l = np.zeros(self.num_ags)
        for i, ag in enumerate(self.ags):
            codis_l[i] = ag.codistillation()

        return codis_l

    def train_codis(self):
        for ag in self.ags:
            ag.calc_codis_data(self.shared_data)
        for ag in self.ags:
            ag.set_codis_data(self.shared_data)

        train_l = np.zeros(self.num_ags)
        train_a = np.zeros(self.num_ags)
        codis_l = np.zeros(self.num_ags)
        for i, ag in enumerate(self.ags):
            train_l[i], train_a[i], codis_l[i] = ag.train_codis()
        return train_l, train_a, codis_l

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
        codis_loss = []
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
                if self.topology not in ConnectionGraph.fixed_topologies:
                    self.set_connection()

                if self.codis_after_train:
                    train_l, train_a = self.train()
                    for _ in range(self.codis_iter):
                        codis_l = self.codistillation()
                    #codis_l = self.codistillation()
                else:
                    train_l, train_a, codis_l = self.train_codis()
                val_l, val_a = self.validation()

                train_loss.append(train_l)
                train_acc.append(train_a)
                codis_loss.append(codis_l)
                valid_loss.append(val_l)
                valid_acc.append(val_a)
                clock_time.append(time.time() - start_time)

                if (epoch + 1) % self.show_step_num == 0:
                    log("epoch {:>3}".format(epoch))
                    log("  loss(train): " + Logger.strarray(train_l))
                    log("  acc.(train): " + Logger.strarray(train_a))
                    log("  codist_loss: " + Logger.strarray(codis_l))
                    log("  loss(valid): " + Logger.strarray(val_l))
                    log("  acc.(valid): " + Logger.strarray(val_a))
                # tmp data
                results["data"] = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "codis_loss": codis_loss,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_acc,
                    "time" : clock_time
                }
                results["logs"] = log
                PickleUtil.save_data(results, "debug/.running")
        except KeyboardInterrupt:
            log("stopped {:>3}".format(epoch))

        results["data"] = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "codis_loss": codis_loss,
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

