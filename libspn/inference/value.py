# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from types import MappingProxyType
from abc import ABC

from libspn import GaussianLeaf
from libspn.graph.algorithms import compute_graph_up, compute_graph_up_dynamic
from libspn.graph.node import DynamicVarNode, DynamicInterface
from libspn.inference.type import InferenceType


class BaseValue(ABC):
    """Assembles TF operations computing the values of nodes of the SPN during
    an upwards pass. The value can be either an SPN value (marginal inference)
    or an MPN value (MPE inference) or a mixture of both.

    Args:
        inference_type (InferenceType): Determines the type of inference that
            should be used. If set to ``None``, the inference type is specified
            by the ``inference_type`` flag of the node. If set to ``MARGINAL``,
            marginal inference will be used for all nodes. If set to ``MPE``,
            MPE inference will be used for all nodes.
    """

    def __init__(self, inference_type=None):
        self._inference_type = inference_type
        self._values = {}

    @property
    def values(self):
        """dict: A dictionary of ``Tensor`` indexed by the SPN node containing
        operations computing value for each node."""
        return MappingProxyType(self._values)


class Value(BaseValue):

    def get_value(self, root):
        """Assemble a TF operation computing the values of nodes of the SPN
        rooted in ``root``.

        Returns the operation computing the value for the ``root``. Operations
        computing values for other nodes can be obtained using :obj:`values`.

        Args:
            root (Node): The root node of the SPN graph.

        Returns:
            Tensor: A tensor of shape ``[None, num_outputs]``, where the first
            dimension corresponds to the batch size.
        """
        def fun(node, *args):
            with tf.name_scope(node.name):
                if (self._inference_type == InferenceType.MARGINAL
                    or (self._inference_type is None and
                        node.inference_type == InferenceType.MARGINAL)):
                    return node._compute_value(*args)
                else:
                    return node._compute_mpe_value(*args)

        self._values = {}
        with tf.name_scope("Value"):
            return compute_graph_up(root, val_fun=fun,
                                    all_values=self._values)


class LogValue(BaseValue):

    def get_value(self, root):
        """Assemble TF operations computing the log values of nodes of the SPN
        rooted in ``root``.

        Returns the operation computing the log value for the ``root``.
        Operations computing log values for other nodes can be obtained using
        :obj:`values`.

        Args:
            root: Root node of the SPN.

        Returns:
            Tensor: A tensor of shape ``[None, num_outputs]``, where the first
            dimension corresponds to the batch size.
        """
        def fun(node, *args):
            with tf.name_scope(node.name):
                if (self._inference_type == InferenceType.MARGINAL
                    or (self._inference_type is None and
                        node.inference_type == InferenceType.MARGINAL)):
                    return node._compute_log_value(*args)
                else:
                    return node._compute_log_mpe_value(*args)

        self._values = {}
        with tf.name_scope("LogValue"):
            return compute_graph_up(root, val_fun=fun,
                                    all_values=self._values)


class DynamicValue(BaseValue):

    def get_value(self, root, return_sequence=False, sequence_lens=None):
        """Assemble a TF operation computing the values of nodes of the SPN
        rooted in ``root``.

        Returns the operation computing the value for the ``root``. Operations
        computing values for other nodes can be obtained using :obj:`values`.

        Args:
            root (Node): The root node of the SPN graph.

        Returns:
            Tensor: A tensor of shape ``[None, num_outputs]``, where the first
            dimension corresponds to the batch size.
        """
        if sequence_lens is not None:
            time0 = root.get_maxlen() - sequence_lens

        def template_val_fun(step, interface_value_map):
            def fun(node, *args):
                with tf.name_scope(node.name):
                    kwargs = {}
                    # TODO the below is not really well designed, but the temporal nodes should
                    # somehow have access to the step (and batch size in the case of a dynamic
                    # interface)
                    if isinstance(node, GaussianLeaf) and node.is_dynamic:
                        kwargs['step'] = step
                    if isinstance(node, DynamicVarNode):
                        kwargs['step'] = step
                    if isinstance(node, DynamicInterface):
                        args = [interface_value_map[node]] if node in interface_value_map else []
                        kwargs['step'] = step
                    if (self._inference_type == InferenceType.MARGINAL
                        or (self._inference_type is None and
                            node.inference_type == InferenceType.MARGINAL)):
                        val = node._compute_value(*args, **kwargs)
                    else:
                        val = node._compute_mpe_value(*args, **kwargs)
                    if sequence_lens is not None and not node.is_param:
                        # If we have sequences with possibly different lengths
                        ones = tf.ones_like(val)
                        if node.interface_head:
                            # In case we have sequences with same length
                            return tf.where(tf.less_equal(step, time0), ones, val)
                        return tf.where(tf.less(step, time0), ones, val)
                    if node.interface_head:
                        # In case we have sequences with same length
                        return tf.cond(tf.equal(step, 0), lambda: tf.ones_like(val), lambda: val)
                    return val
            return fun

        self._values = {}
        with tf.name_scope("Value"):
            top_val, top_per_step = compute_graph_up_dynamic(
                root=root, val_fun_step=template_val_fun, all_values=self._values,
                interface_init=tf.ones
            )

        if return_sequence:
            # Optionally return top value per time step
            return top_val, top_per_step
        return top_val


class DynamicLogValue(BaseValue):

    def get_value(self, root, return_sequence=False, sequence_lens=None):
        """Assemble a TF operation computing the values of nodes of the SPN
        rooted in ``root``.

        Returns the operation computing the value for the ``root``. Operations
        computing values for other nodes can be obtained using :obj:`values`.

        Args:
            root (Node): The root node of the SPN graph.

        Returns:
            Tensor: A tensor of shape ``[None, num_outputs]``, where the first
            dimension corresponds to the batch size.
        """
        if sequence_lens is not None:
            time0 = root.get_maxlen() - sequence_lens

        def val_fun_step(step, interface_value_map):
            def fun(node, *args):
                with tf.name_scope(node.name):
                    kwargs = {}
                    if isinstance(node, GaussianLeaf) and node.is_dynamic:
                        kwargs['step'] = step
                    if isinstance(node, DynamicVarNode):
                        kwargs['step'] = step
                    if isinstance(node, DynamicInterface):
                        args = [interface_value_map[node]] if node in interface_value_map else []
                        kwargs['step'] = step
                    if (self._inference_type == InferenceType.MARGINAL
                        or (self._inference_type is None and
                            node.inference_type == InferenceType.MARGINAL)):
                        val = node._compute_log_value(*args, **kwargs)
                    else:
                        val = node._compute_log_mpe_value(*args, **kwargs)
                    if sequence_lens is not None and not node.is_param:
                        # If we have sequences with possibly different lengths
                        zeros = tf.zeros_like(val)
                        if node.interface_head:
                            # In case we have sequences with same length
                            return tf.where(tf.less_equal(step, time0), zeros, val)
                        return tf.where(tf.less(step, time0), zeros, val)
                    if node.interface_head:
                        # In case we have sequences with same length
                        return tf.cond(tf.equal(step, 0), lambda: tf.zeros_like(val), lambda: val)
                    return val
            return fun

        self._values = {}
        with tf.name_scope("LogValue"):
            top_val, top_per_step = compute_graph_up_dynamic(
                root=root, val_fun_step=val_fun_step, all_values=self._values,
                interface_init=tf.zeros)

        if return_sequence:
            # Optionally return top value per time step
            return top_val, top_per_step
        return top_val
