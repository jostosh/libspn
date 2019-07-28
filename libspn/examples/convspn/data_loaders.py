import os
from os import path as opth

import numpy as np
from sklearn.datasets import olivetti_faces
import tensorflow as tf

tfk = tf.keras


def load_fashion_mnist(args):
    (train_x, train_y), (test_x, test_y) = tfk.datasets.fashion_mnist.load_data()
    train_x = np.reshape(train_x / 255., (-1, 28, 28, 1))
    test_x = np.reshape(test_x / 255., (-1, 28, 28, 1))
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()
    return test_x, test_y, train_x, train_y


def load_cifar10(args):
    (train_x, train_y), (test_x, test_y) = tfk.datasets.cifar10.load_data()
    train_x = train_x / 255.
    test_x = test_x / 255.
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()
    return test_x, test_y, train_x, train_y


def load_mnist(args):
    (train_x, train_y), (test_x, test_y) = tfk.datasets.mnist.load_data()
    train_x = np.reshape(train_x / 255., (-1, 28, 28, 1))
    test_x = np.reshape(test_x / 255., (-1, 28, 28, 1))
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()

    if args.discrete:
        train_x = np.greater(train_x, 20 / 256).astype(np.int32)
        test_x = np.greater(test_x, 20 / 256).astype(np.int32)

    return test_x, test_y, train_x, train_y


def load_olivetti(args):
    bunch = olivetti_faces.fetch_olivetti_faces()
    x, y = np.expand_dims(bunch.images, axis=-1), bunch.target
    train_x, train_y, test_x, test_y = [], [], [], []
    for label in range(max(y) + 1):
        x_class = x[y == label]
        y_class = [label] * len(x_class)
        # print(label, len(x_class))
        test_size = min(30, len(x_class) // 3)
        train_x.extend(x_class[:-test_size])
        train_y.extend(y_class[:-test_size])
        test_x.extend(x_class[-test_size:])
        test_y.extend(y_class[-test_size:])
    train_x, test_x, train_y, test_y = np.asarray(train_x), np.asarray(test_x), \
                                       np.asarray(train_y), np.asarray(test_y)
    if args.normalize_data:
        return test_x * 255, test_y, train_x * 255, train_y
    return test_x, test_y, train_x, train_y


def load_caltech(args):
    base = opth.expanduser("~/datasets/caltech")
    class_dirs = [d for d in os.listdir(base) if opth.isdir(opth.join(base, d))]
    train_x, train_y, test_x, test_y = [], [], [], []
    for label, cd in enumerate(class_dirs):
        dirpath = opth.join(base, cd)
        fnms = os.listdir(dirpath)
        x = [np.loadtxt(opth.join(dirpath, fnm)) for fnm in fnms]
        y = [label] * len(fnms)
        test_size = min(50, len(fnms) // 3)
        train_x.extend(x[:-test_size])
        train_y.extend(y[:-test_size])
        test_x.extend(x[-test_size:])
        test_y.extend(y[-test_size:])

    train_x = np.expand_dims(np.asarray(train_x), -1)
    test_x = np.expand_dims(np.asarray(test_x), -1)
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)

    if not args.normalize_data:
        train_x /= 255.
        test_x /= 255.

    return test_x, test_y, train_x, train_y