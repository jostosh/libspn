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
class ContVars(VarNode):
    """A node representing a vector of continuous random variables.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of variables in the vector.
        name (str): Name of the node
    """

    def __init__(self, feed=None, num_vars=1, name="ContVars"):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        self._num_vars = num_vars
        super().__init__(feed, name)

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        super().deserialize(data)

    def _create_placeholder(self):
        """Create a placeholder that will be used to feed this variable when
        no other feed is available.

        Returns:
            Tensor: A TF placeholder of shape ``[None, num_vars]``, where the
            first dimension corresponds to the batch size.
        """
        return tf.placeholder(conf.dtype, [None, self._num_vars])

    def _compute_out_size(self):
        return self._num_vars

    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars)]

    def _compute_value(self):
        # We used identity, since this way we can feed and fetch this node
        # and there is an operation in TensorBoard even if the internal
        # placeholder is not used for feeding.
        return tf.identity(self._feed)

    def _compute_mpe_state(self, counts):
        return counts


@register_serializable
class DynamicContVars(DynamicVarNode):
    """A node representing a vector of continuous random variables.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of variables in the vector.
        name (str): Name of the node
    """

    def __init__(self, max_steps, time_major=True, feed=None, num_vars=1, name="ContVars"):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        self._num_vars = num_vars
        super().__init__(max_steps=max_steps, feed=feed, name=name, time_major=time_major)

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['max_steps'] = self._max_steps
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._max_steps = data['max_steps']
        super().deserialize(data)

    def _create_placeholder(self):
        """Create a placeholder that will be used to feed this variable when
        no other feed is available.

        Returns:
            Tensor: A TF placeholder of shape ``[None, num_vars]``, where the
            first dimension corresponds to the batch size.
        """
        return tf.placeholder(conf.dtype, [self._max_steps, None, self._num_vars])

    def _compute_out_size(self):
        return self._num_vars

    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars)]

    def _compute_array(self):
        array = tf.TensorArray(dtype=conf.dtype, size=self._max_steps).unstack(value=self._feed)
        return array

    def _compute_value(self, step):
        # We used identity, since this way we can feed and fetch this node
        # and there is an operation in TensorBoard even if the internal
        # placeholder is not used for feeding.
        array = self._compute_array()
        return array.read(step)

    def _compute_mpe_state(self, counts):
        return counts
