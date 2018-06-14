# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from types import MappingProxyType
import abc
from libspn.graph.algorithms import compute_graph_up, compute_graph_up_dynamic
from libspn.graph.node import DynamicVarNode, DynamicInterface
from libspn.graph.distribution import GaussianLeaf
from libspn.inference.type import InferenceType
from libspn import utils
from libspn.graph.basesum import BaseSum


class BaseValue(abc.ABC):
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

    def __init__(self, inference_type=None, log=True, name="Value", dropconnect_keep_prob=None, 
                 dropout_keep_prob=None):
        if inference_type not in [None, InferenceType.MARGINAL, InferenceType.MPE]:
            raise ValueError(
                "Inference type must either None or libspn.InferenceType.MARGINAL "
                "or libspn.InferenceType.MPE, got {}.".format(inference_type))
        self._inference_type = inference_type
        self._values = {}
        self._dropconnect_keep_prob = dropconnect_keep_prob
        self._dropout_keep_prob = dropout_keep_prob
        self._name = name
        self._log = log

    @property
    def log(self):
        return self._log
    
    @property
    def values(self):
        """dict: A dictionary of ``Tensor`` indexed by the SPN node containing
        operations computing value for each node."""
        return MappingProxyType(self._values)

    def _get_node_kwargs(self, node):
        if isinstance(node, BaseSum):
            return dict(dropout_keep_prob=self._dropout_keep_prob,
                        dropconnect_keep_prob=self._dropconnect_keep_prob)
        return dict()

    def _value_fn(self):
        def fun(node, *args):
            with tf.name_scope(node.name):
                kwargs = self._get_node_kwargs(node)
                inference_type = self._inference_type or node.inference_type
                if inference_type == InferenceType.MARGINAL:
                    if self._log:
                        return node._compute_log_value(*args, **kwargs)
                    else:
                        return node._compute_value(*args, **kwargs)
                elif inference_type == InferenceType.MPE:
                    if self._log:
                        return node._compute_log_mpe_value(*args, **kwargs)
                    else:
                        return node._compute_mpe_value(*args, **kwargs)
                else:
                    raise TypeError("Inference type must be either marginal or MPE.")
        return fun

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

        self._values = {}
        with tf.name_scope(self._name):
            return compute_graph_up(root, val_fun=self._value_fn(), all_values=self._values)


class Value(BaseValue):

    def __init__(self, inference_type=None, dropconnect_keep_prob=None, dropout_keep_prob=None, 
                 name="Value"):
        super().__init__(inference_type=inference_type, name=name, log=False, 
                         dropconnect_keep_prob=dropconnect_keep_prob, 
                         dropout_keep_prob=dropout_keep_prob)


class LogValue(BaseValue):

    def __init__(self, inference_type=None, dropconnect_keep_prob=None, dropout_keep_prob=None, 
                 name="LogValue"):
        super().__init__(inference_type=inference_type, name=name, log=True, 
                         dropconnect_keep_prob=dropconnect_keep_prob, 
                         dropout_keep_prob=dropout_keep_prob)


class DynamicBaseValue(BaseValue, abc.ABC):

    @utils.lru_cache
    def read_node_val(self, node, t):
        return self._values[node].read(t)

    def _get_node_kwargs(self, node, step, interface_value_map):
        kwargs = super()._get_node_kwargs(node)
        if isinstance(node, (DynamicVarNode, DynamicInterface)) or \
                (isinstance(node, GaussianLeaf) and node.is_dynamic):
            kwargs['step'] = step
        return kwargs
    
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

        no_evidence_fn = tf.zeros_like if self._log else tf.ones_like

        def template_val_fun(step, interface_value_map):
            def fun(node, *args):
                with tf.name_scope(node.name):
                    kwargs = {}
                    # TODO the below is not really well designed, but the temporal nodes should
                    # somehow have access to the step (and batch size in the case of a dynamic
                    # interface)
                    if isinstance(node, DynamicInterface):
                        args = [interface_value_map[node]] if node in interface_value_map else []
                    kwargs = self._get_node_kwargs(node, step, interface_value_map)    
                    inference_type = self._inference_type or node.inference_type
                    if inference_type == InferenceType.MARGINAL:
                        if self._log:
                            val = node._compute_log_value(*args, **kwargs)
                        else:
                            val = node._compute_value(*args, **kwargs)
                    elif inference_type == InferenceType.MPE:
                        if self._log:
                            val = node._compute_log_mpe_value(*args, **kwargs)
                        else:
                            val = node._compute_mpe_value(*args, **kwargs)
                    else:
                        raise TypeError("Inference type must be either MARGINAL or MPE")
                    if sequence_lens is not None and not node.is_param:
                        # If we have sequences with possibly different lengths
                        no_evidence = no_evidence_fn(val)
                        if node.interface_head:
                            # In case we have sequences with same length
                            return tf.where(tf.less_equal(step, time0), no_evidence, val)
                        return tf.where(tf.less(step, time0), no_evidence, val)
                    if node.interface_head:
                        # In case we have sequences with same length
                        return tf.cond(tf.equal(step, 0), lambda: no_evidence_fn(val), lambda: val)
                    return val
            return fun

        self._values = {}
        with tf.name_scope(self._name):
            top_val, top_per_step = compute_graph_up_dynamic(
                root=root, val_fun_step=template_val_fun, all_values=self._values,
                interface_init=tf.ones
            )

        if return_sequence:
            # Optionally return top value per time step
            return top_val, top_per_step
        return top_val


class DynamicValue(DynamicBaseValue):

    def __init__(self, inference_type=None, dropout_keep_prob=None, dropconnect_keep_prob=None, 
                 name="DynamicValue"):
        super().__init__(inference_type=inference_type, dropout_keep_prob=dropout_keep_prob, 
                         dropconnect_keep_prob=dropconnect_keep_prob, name=name, log=False)


class DynamicLogValue(DynamicBaseValue):

    def __init__(self, inference_type=None, dropout_keep_prob=None, dropconnect_keep_prob=None, 
                 name="DynamicLogValue"):
        super().__init__(inference_type=inference_type, dropout_keep_prob=dropout_keep_prob, 
                         dropconnect_keep_prob=dropconnect_keep_prob, name=name, log=True)


