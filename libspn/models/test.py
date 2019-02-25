# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.models.model import Model
from libspn.graph.product import Product
from libspn.graph.ivs import IVs
from libspn.graph.sum import Sum
from libspn import conf
from libspn import utils
import numpy as np
import tensorflow as tf


class Poon11NaiveMixtureModel(Model):

    """A simple naive Bayes mixture from the Poon&Domingos'11 paper.

    The model is only used for testing. The weights of the model are initialized
    to specific values, for which various qualities are calculated.
    """

    def __init__(self):
        super().__init__()
        self._ivs = None

    @utils.docinherit(Model)
    def serialize(save_param_vals=True, sess=None):
        raise NotImplementedError("Serialization not implemented")

    @utils.docinherit(Model)
    def deserialize(self, data, load_param_vals=True, sess=None):
        raise NotImplementedError("Serialization not implemented")

    @property
    def ivs(self):
        """IVs: The IVs with the input variables of the model."""
        return self._ivs

    @property
    def true_mpe_state(self):
        """The true MPE state for the SPN."""
        return np.array([1, 0])

    @property
    def true_values(self):
        """The true values of the SPN for the :meth:`feed`."""
        return np.array([[1.0],
                         [0.75],
                         [0.25],
                         [0.31],
                         [0.228],
                         [0.082],
                         [0.69],
                         [0.522],
                         [0.168]], dtype=conf.dtype.as_numpy_dtype)

    @property
    def true_mpe_values(self):
        """The true MPE values of the SPN for the :meth:`feed`."""
        return np.array([[0.216],
                         [0.216],
                         [0.09],
                         [0.14],
                         [0.14],
                         [0.06],
                         [0.216],
                         [0.216],
                         [0.09]], dtype=conf.dtype.as_numpy_dtype)

    @property
    def feed(self):
        """Feed containing all possible values of the input variables."""
        values = np.arange(-1, 2)
        points = np.array(np.meshgrid(*[values for i in range(2)])).T
        return points.reshape(-1, points.shape[-1])

    @utils.docinherit(Model)
    def build(self):
        # Inputs
        self._ivs = IVs(num_vars=2, num_vals=2, name="IVs")
        # Input mixtures
        s11 = Sum((self._ivs, [0, 1]), name="Sum1.1")
        s11.generate_weights(tf.initializers.constant([0.4, 0.6]))
        s12 = Sum((self._ivs, [0, 1]), name="Sum1.2")
        s12.generate_weights(tf.initializers.constant([0.1, 0.9]))
        s21 = Sum((self._ivs, [2, 3]), name="Sum2.1")
        s21.generate_weights(tf.initializers.constant([0.7, 0.3]))
        s22 = Sum((self._ivs, [2, 3]), name="Sum2.2")
        s22.generate_weights(tf.initializers.constant([0.8, 0.2]))
        # Components
        p1 = Product(s11, s21, name="Comp1")
        p2 = Product(s11, s22, name="Comp2")
        p3 = Product(s12, s22, name="Comp3")
        # Mixing components
        self._root = Sum(p1, p2, p3, name="Mixture")
        self._root.generate_weights(tf.initializers.constant([0.5, 0.2, 0.3]))
        return self._root
