#!/usr/bin/env python3

import cifar10
import mnist
import fashion

from utils import PickleUtil

def main():
    # mnist
    m_param = {
        "num_labels_ag": 2,
        "num_epoch": 1000,
        "topology": "onehop",
        #"start_m": 3,
        "nn_model": "model_0",
        "learning_rate": 0.1,
        "averaging_rate": 0.1,
        "show_step_num": 50,
    }

    # fashion
    f_param = {
        "num_labels_ag": 2,
        "num_epoch": 1000,
        "nn_model": "model_0",
        "topology": "onehop",
        #"start_m": start_m,
        "learning_rate": 0.1,
        "averaging_rate": 0.1,
        "show_step_num": 50,
    }

    # cifar10
    c_param = {
        "num_labels_ag": 2,
        "num_epoch": 2000,
        "num_train_data": 40000,
        "minibatchsize": 50,
        "minibatchsize_shared": 50,
        "nn_model": "model_0",
        "topology": "onehop",
        #"start_m": 3,
        "learning_rate": 0.1,
        "averaging_rate": 0.1,
        "dropout_cnn": 0.2,
        "show_step_num": 50,
    }

    for lib, prm, fname in [
            (mnist, m_param, "results/mnist.dat"),
            (fashion, f_param, "results/fashion.dat"),
            (cifar10, c_param, "results/cifar10.dat"),
            ]:
        dtype = "float32"
        lib.set_tf_config(dtype)

        sim = lib.Simulator(prm)
        rslt = sim.run()
        PickleUtil.save_data(rslt, fname)
        print("File saved:", fname)

    return sim

if __name__ == "__main__":
    sim = main()

