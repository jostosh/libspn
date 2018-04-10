# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import VarNode, DynamicVarNode
from libspn import conf
from libspn.utils.serialization import register_serializable


@register_serializable
class IVs(VarNode):
    """A node representing multiple random variables in the form of indicator
    variables. Each random variable is assumed to take the same number of
    possible values ``[0, 1, ..., num_vals-1]``. If the value of the random
    variable is negative (e.g. -1), all indicators will be set to 1.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of random variables.
        num_vals (int): Number of values of each random variable.
        name (str): Name of the node
    """

    def __init__(self, feed=None, num_vars=1, num_vals=2, name="IVs"):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        if not isinstance(num_vals, int) or num_vals < 2:
            raise ValueError("num_vals must be >1")
        self._num_vars = num_vars
        self._num_vals = num_vals
        super().__init__(feed, name)

    @property
    def num_vars(self):
        """int: Number of random variables of the IVs."""
        return self._num_vars

    @property
    def num_vals(self):
        """int: Number of values of each random variable."""
        return self._num_vals

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_vals'] = self._num_vals
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_vals = data['num_vals']
        super().deserialize(data)

    def _create_placeholder(self):
        """Create a placeholder that will be used to feed this variable when
        no other feed is available.

        Returns:
            Tensor: An integer TF placeholder of shape ``[None, num_vars]``,
            where the first dimension corresponds to the batch size.
        """
        return tf.placeholder(tf.int32, [None, self._num_vars])

    def _compute_out_size(self):
        return self._num_vars * self._num_vals

    def _compute_scope(self):
        return [Scope(self, i)
                for i in range(self._num_vars)
                for _ in range(self._num_vals)]

    def _compute_value(self):
        """Assemble the TF operations computing the output value of the node
        for a normal upwards pass.

        This function converts the integer inputs to indicators.

        Returns:
            Tensor: A tensor of shape ``[None, num_vars*num_vals]``, where the
            first dimension corresponds to the batch size.
        """
        # The output type has to be conf.dtype otherwise MatMul will
        # complain about being unable to mix types
        oh = tf.one_hot(self._feed, self._num_vals, dtype=conf.dtype)
        # Detect negative input values and convert them to all IVs equal to 1
        neg = tf.expand_dims(tf.cast(tf.less(self._feed, 0), dtype=conf.dtype), dim=-1)
        oh = tf.add(oh, neg)
        # Reshape
        return tf.reshape(oh, [-1, self._num_vars * self._num_vals])

    def _compute_mpe_state(self, counts):
        r = tf.reshape(counts, (-1, self._num_vars, self._num_vals))

        out = tf.argmax(r, dimension=2)

        # Check whether we have a sequence dimension in input and reshape output accordingly
        if len(counts.shape) == 3:
            shape_tensor = tf.shape(counts)
            sequence_dim = shape_tensor[0]
            batch_dim = shape_tensor[1]
            out = tf.reshape(out, (sequence_dim, batch_dim, self._num_vars))

        return out


@register_serializable
class DynamicIVs(DynamicVarNode):
    """A node representing multiple random variables in the form of indicator
    variables. Each random variable is assumed to take the same number of
    possible values ``[0, 1, ..., num_vals-1]``. If the value of the random
    variable is negative (e.g. -1), all indicators will be set to 1.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of random variables.
        num_vals (int): Number of values of each random variable.
        name (str): Name of the node
    """

    def __init__(self, max_steps, feed=None, num_vars=1, num_vals=2, name="IVs", time_major=True):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        if not isinstance(num_vals, int) or num_vals < 2:
            raise ValueError("num_vals must be >1")
        self._num_vars = num_vars
        self._num_vals = num_vals
        super().__init__(max_steps=max_steps, time_major=time_major, feed=feed, name=name)

    @property
    def num_vars(self):
        """int: Number of random variables of the IVs."""
        return self._num_vars

    @property
    def num_vals(self):
        """int: Number of values of each random variable."""
        return self._num_vals

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_vals'] = self._num_vals
        data['max_steps'] = self._max_steps
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_vals = data['num_vals']
        self._max_steps = data['max_steps']
        super().deserialize(data)

    def _create_placeholder(self):
        """Create a placeholder that will be used to feed this variable when
        no other feed is available.

        Returns:
            Tensor: An integer TF placeholder of shape ``[None, num_vars]``,
            where the first dimension corresponds to the batch size.
        """
        return tf.placeholder(tf.int32, [self._max_steps, None, self._num_vars])

    def _compute_out_size(self):
        return self._num_vars * self._num_vals

    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars) for _ in range(self._num_vals)]

    def _compute_dense_array(self):
        # The output type has to be conf.dtype otherwise MatMul will
        # complain about being unable to mix types
        oh = tf.one_hot(self._feed, self._num_vals, dtype=conf.dtype)
        # Detect negative input values and convert them to all IVs equal to 1
        neg = tf.expand_dims(tf.cast(tf.less(self._feed, 0), dtype=conf.dtype), dim=-1)
        oh = tf.add(oh, neg)
        # Reshape
        dense = tf.reshape(oh, [self._max_steps, -1, self._num_vars * self._num_vals])
        array = tf.TensorArray(dtype=conf.dtype, size=self._max_steps).unstack(value=dense)
        return array

    def _compute_value(self, step):
        """Assemble the TF operations computing the output value of the node
        for a normal upwards pass.

        This function converts the integer inputs to indicators.

        Returns:
            Tensor: A tensor of shape ``[None, num_vars*num_vals]``, where the
            first dimension corresponds to the batch size.
            :param **kwargs:
        """
        dense_ivs = self._compute_dense_array()
        return dense_ivs.read(step)

    def _compute_mpe_state(self, counts):
        # TODO not thought about this yet
        if len(counts.shape) == 2:
            r = tf.reshape(counts, (-1, self._num_vars, self._num_vals))
            return tf.argmax(r, dimension=2)
        elif len(counts.shape) == 3:
            r = tf.reshape(counts, (self._max_steps, -1, self._num_vars, self._num_vals))
            return tf.argmax(r, axis=-1)
        else:
            raise ValueError("Counts should be a tensor of either rank 2 or 3.")
