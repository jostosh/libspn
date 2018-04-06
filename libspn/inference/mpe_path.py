# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from types import MappingProxyType
import tensorflow as tf

from libspn import Product
from libspn.inference.value import Value, LogValue, DynamicValue, DynamicLogValue
from libspn.graph.algorithms import compute_graph_up_down, compute_graph_up_down_dynamic, \
    get_batch_size, get_max_steps
from libspn.utils.defaultordereddict import DefaultOrderedDict
import libspn.conf as conf


def tensor_array_factory(max_len):
    def factory(node):
        return tf.TensorArray(
            size=max_len, dtype=conf.dtype, clear_after_read=False, name=node.name + "_TensorArray")
    return factory


class MPEPath:
    """Assembles TF operations computing the branch counts for the MPE downward
    path through the SPN. It computes the number of times each branch was
    traveled by a complete subcircuit determined by the MPE value of the latent
    variables in the model.

    Args:
        value (Value or LogValue): Pre-computed SPN values.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``value`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``value`` is given.
    """

    def __init__(self, value=None, value_inference_type=None, log=True, add_random=None,
                 use_unweighted=False, dynamic=False):
        self._counts = {} #if not dynamic else DefaultOrderedDict(
            # default_factory=tensor_array_factory(maxlen), pass_key=True)
        self._log = log
        self._add_random = add_random
        self._use_unweighted = use_unweighted
        self._dynamic = dynamic
        # Create internal value generator
        if value is None:
            if dynamic:
                if log:
                    self._value = DynamicLogValue(value_inference_type)
                else:
                    self._value = DynamicValue(value_inference_type)
            else:
                if log:
                    self._value = LogValue(value_inference_type)
                else:
                    self._value = Value(value_inference_type)
        else:
            self._value = value

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._value

    @property
    def counts(self):
        """dict: Dictionary indexed by node, where each value is a lists of
        tensors computing the branch counts for the inputs of the node."""
        return MappingProxyType(self._counts)

    def _get_mpe_path_feedforward(self, root):
        """Assemble TF operations computing the branch counts for the MPE
        downward path through the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        def down_fun(node, parent_vals):
            # Sum up all parent vals
            if len(parent_vals) > 1:
                summed = tf.add_n(parent_vals, name=node.name + "_add")
            else:
                summed = parent_vals[0]
            self._counts[node] = summed
            if node.is_op:
                # Compute for inputs
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_mpe_path(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs],
                            add_random=self._add_random,
                            use_unweighted=self._use_unweighted)
                    else:
                        return node._compute_mpe_path(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs],
                            add_random=self._add_random,
                            use_unweighted=self._use_unweighted)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("MPEPath"):
            # Compute the tensor to feed to the root node
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing counts
            self._counts = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)

    def get_mpe_path(self, root):
        if not self._dynamic:
            self._get_mpe_path_feedforward(root)
        else:
            self._get_mpe_path_dynamic(root)

    def _get_mpe_path_dynamic(self, root):
        """Assemble TF operations computing the branch counts for the MPE
        downward path through the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """

        self._counts = DefaultOrderedDict(
            default_factory=tensor_array_factory(get_max_steps(root)), pass_key=True)

        def combine_parents_fun_time(t):
            def combine_parents_fun(node, parent_vals):
                # Sum up all parent vals
                def accumulate():
                    if len(parent_vals) > 1:
                        summed = tf.add_n(parent_vals, name=node.name + "_add")
                    else:
                        summed = parent_vals[0]
                    return summed

                if node.is_op and node.interface_head:
                    return tf.cond(tf.equal(t, 0),
                                   lambda: tf.zeros_like(parent_vals[0]), accumulate)
                return accumulate()
            return combine_parents_fun

        def down_fun_time(t):
            def down_fun(node, summed):
                # # Sum up all parent vals
                # if t == 0 and node.is_op and node.interface_head:
                #     summed = tf.zeros_like(parent_vals[0])
                # else:
                #     if len(parent_vals) > 1:
                #         summed = tf.accumulate_n(parent_vals, name=node.name + "_add")
                #     else:
                #         summed = parent_vals[0]
                # with tf.name_scope("WriteCountsToArray"):
                #     self._counts[node] = self._counts[node].write(t, summed)
                if node.is_op:
                    # Compute for inputs
                    with tf.name_scope(node.name):
                        # if node.is_interface:
                        #     inputs = node.source.inputs
                        #     time = t - 1
                        # else:
                        #     inputs = node.inputs
                        #     time = t

                        if self._log:
                            return node._compute_log_mpe_path(
                                summed, *[self._value.values[i.node].read(t)
                                          if i else None
                                          for i in node.inputs],
                                add_random=self._add_random,
                                use_unweighted=self._use_unweighted)
                        else:
                            return node._compute_mpe_path(
                                summed, *[self._value.values[i.node].read(t)
                                          if i else None
                                          for i in node.inputs],
                                add_random=self._add_random,
                                use_unweighted=self._use_unweighted)
            return down_fun

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("MPEPath"):
            # Compute the tensor to feed to the root node
            out_size = root.get_out_size()
            out_size = (out_size,) if isinstance(out_size, int) else out_size
            graph_input_end = tf.ones(shape=(get_batch_size(root),) + out_size, dtype=conf.dtype)
            graph_input_default = tf.zeros_like(graph_input_end)

            # Traverse the graph computing counts
            self._counts = compute_graph_up_down_dynamic(
                root, down_fun_time=down_fun_time, graph_input_end=graph_input_end,
                graph_input_default=graph_input_default,
                combine_parents_fun_time=combine_parents_fun_time)

            for node in self._counts:
                self._counts[node] = tf.reduce_sum(
                    self._counts[node].stack(), axis=0, name=node.name + "CountsTotalPerBatch")
