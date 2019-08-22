import os
from os import path as opth

import numpy as np
from sklearn.datasets import olivetti_faces
import tensorflow as tf
from skimage.io import imsave

tfk = tf.keras

def normalize_batch_wise(train_x, test_x):
    mu, sigma = np.mean(train_x, axis=0, keepdims=True), np.std(train_x, axis=0, keepdims=True)

    def _z(x):
        return (x - mu) / (sigma + 1.0)

    return _z(train_x), _z(test_x)

def load_fashion_mnist(args):
    (train_x, train_y), (test_x, test_y) = tfk.datasets.fashion_mnist.load_data()
    train_x = np.reshape(train_x, (-1, 28, 28, 1))
    test_x = np.reshape(test_x, (-1, 28, 28, 1))
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()

    if not args.normalize_data:
        train_x /= 255.
        test_x /= 255.

    if args.normalize_batch_wise:
        train_x, test_x = normalize_batch_wise(train_x, test_x)

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
    train_x = np.reshape(train_x, (-1, 28, 28, 1))
    test_x = np.reshape(test_x, (-1, 28, 28, 1))
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()

    if args.discrete:
        train_x = np.greater(train_x, 20 / 256).astype(np.int32)
        test_x = np.greater(test_x, 20 / 256).astype(np.int32)

    # if not args.normalize_data:

    if args.normalize_batch_wise:
        train_x, test_x = normalize_batch_wise(train_x, test_x)
    elif not args.normalize_data:
        train_x = train_x / 255.
        test_x = test_x / 255.

    return test_x, test_y, train_x, train_y


def load_olivetti(args):
    bunch = olivetti_faces.fetch_olivetti_faces()
    x, y = np.expand_dims(bunch.images, axis=-1), bunch.target
    train_x, train_y, test_x, test_y = [], [], [], []

    x = np.loadtxt("olivetti.raw").transpose().reshape(400, 64, 64, 1).transpose((0, 2, 1, 3)) / 255.

    for i, (xi, yi) in enumerate(zip(x, y)):
        if i < len(x) - 50:
            train_x.append(xi)
            train_y.append(yi)
        else:
            test_x.append(xi)
            test_y.append(yi)

    #
    # for label in range(max(y) + 1):
    #     x_class = x[y == label]
    #     y_class = [label] * len(x_class)
    #     # print(label, len(x_class))
    #     test_size = min(30, len(x_class) // 3)
    #     train_x.extend(x_class[:-test_size])
    #     train_y.extend(y_class[:-test_size])
    #     test_x.extend(x_class[-test_size:])
    #     test_y.extend(y_class[-test_size:])
    train_x, test_x, train_y, test_y = np.asarray(train_x), np.asarray(test_x), \
                                       np.asarray(train_y), np.asarray(test_y)

    if args.normalize_data:
        return test_x * 255, test_y, train_x * 255, train_y
    return test_x, test_y, train_x, train_y


def load_caltech(args):
    base = opth.expanduser("~/datasets/caltech")

    rescale_size = 100
    crop_size = 64
    offset = (rescale_size - crop_size) // 2

    if opth.exists(opth.join(base, 'train_x.npy')) and opth.exists(opth.join(base, 'train_y.npy')) and \
            opth.exists(opth.join(base, 'test_x.npy')) and opth.exists(opth.join(base, 'test_y.npy')):
        train_x = np.load(opth.join(base, 'train_x.npy'))
        test_x = np.load(opth.join(base, 'test_x.npy'))
        train_y = np.load(opth.join(base, 'train_y.npy'))
        test_y = np.load(opth.join(base, 'test_y.npy'))
    else:
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

        # Take the inner 64 x 64 pixels (see spn-user-guide.pdf in Poon and Domingos code)
        train_x = train_x[:, offset:offset + crop_size, offset:offset + crop_size, :]
        test_x = test_x[:, offset:offset + crop_size, offset:offset + crop_size, :]

        np.save(opth.join(base, 'train_x.npy'), train_x)
        np.save(opth.join(base, 'train_y.npy'), train_y)
        np.save(opth.join(base, 'test_x.npy'), test_x)
        np.save(opth.join(base, 'test_y.npy'), test_y)

    if not args.normalize_data:
        train_x /= 255.
        test_x /= 255.

    return test_x, test_y, train_x, train_y