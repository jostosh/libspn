# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from types import MappingProxyType
import tensorflow as tf
from libspn.graph.basesum import BaseSum
from libspn.inference.value import Value, LogValue, DynamicValue, DynamicLogValue
from libspn.graph.algorithms import compute_graph_up_down, compute_graph_up_down_dynamic
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
                 use_unweighted=False, sample=False, sample_prob=None, sample_rank_based=None,
                 dropconnect_keep_prob=None, dropout_keep_prob=None, dynamic=False, 
                 dynamic_reduce_in_loop=False):
        self._true_counts = {}
        self._actual_counts = {}
        self._counts_per_step = {}
        self._log = log
        self._add_random = add_random
        self._use_unweighted = use_unweighted
        self._sample = sample
        self._sample_prob = sample_prob
        self._sample_rank_based = sample_rank_based
        self._dynamic = dynamic
        self._dynamic_reduce_in_loop = dynamic_reduce_in_loop
        # Create internal value generator
        if value is None:
            if dynamic:
                if log:
                    self._value = DynamicLogValue(
                        value_inference_type, dropout_keep_prob=dropout_keep_prob,
                        dropconnect_keep_prob=dropconnect_keep_prob)
                else:
                    self._value = DynamicValue(
                        value_inference_type, dropout_keep_prob=dropout_keep_prob,
                        dropconnect_keep_prob=dropconnect_keep_prob)
            else:
                if log:
                    self._value = LogValue(
                        value_inference_type, dropout_keep_prob=dropout_keep_prob,
                        dropconnect_keep_prob=dropconnect_keep_prob)
                else:
                    self._value = Value(
                        value_inference_type, dropout_keep_prob=dropout_keep_prob,
                        dropconnect_keep_prob=dropconnect_keep_prob)
        else:
            self._value = value
            self._log = value.log()

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._value

    @property
    def counts(self):
        """dict: Dictionary indexed by node, where each value is a list of tensors
        computing the branch counts, based on the true value of the SPN's latent
        variable, for the inputs of the node."""
        return MappingProxyType(self._true_counts)

    @property
    def actual_counts(self):
        """dict: Dictionary indexed by node, where each value is a list of tensors
        computing the branch counts, based on the actual value calculated by the
        SPN, for the inputs of the node."""
        return MappingProxyType(self._actual_counts)

    @property
    def log(self):
        return self._log

    @property
    def counts_per_step(self):
        """dict: Dictionary indexed by node, where each value is a tensor
        that computes the counts with dimensions [time, batch, node] """
        if not self._dynamic:
            raise AttributeError("MPE path was not configured for a dynamic SPN. Maybe you want to "
                                 "use dynamic=True when instantiating.")
        return MappingProxyType(self._counts_per_step)

    def _get_mpe_path_fixed_graph(self, root):
        """Assemble TF operations computing the branch counts for the MPE
        downward path through the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        def down_fun(node, parent_vals):
            # Sum up all parent vals
            parent_vals = [pv for pv in parent_vals if pv is not None]
            if len(parent_vals) > 1:
                summed = tf.add_n(parent_vals, name=node.name + "_add")
            else:
                summed = parent_vals[0]
            self._true_counts[node] = summed
            basesum_kwargs = dict(
                add_random=self._add_random, use_unweighted=self._use_unweighted,
                with_ivs=True, sample=self._sample, sample_prob=self._sample_prob,
                sample_rank_based=self._sample_rank_based)
            if node.is_op:
                kwargs = basesum_kwargs if isinstance(node, BaseSum) else dict()
                # Compute for inputs
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)
                    else:
                        return node._compute_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("TrueMPEPath"):
            # Compute the tensor to feed to the root node
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing counts
            self._true_counts = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)

    def get_mpe_path_actual(self, root):
        """Assemble TF operations computing the actual branch counts for the MPE
        downward path through the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        def down_fun(node, parent_vals):
            # Sum up all parent vals
            parent_vals = [pv for pv in parent_vals if pv is not None]
            if len(parent_vals) > 1:
                summed = tf.add_n(parent_vals, name=node.name + "_add")
            else:
                summed = parent_vals[0]
            self._actual_counts[node] = summed
            basesum_kwargs = dict(
                add_random=self._add_random, use_unweighted=self._use_unweighted,
                with_ivs=False, sample=self._sample, sample_prob=self._sample_prob,
                sample_rank_based=self._sample_rank_based)
            if node.is_op:
                # Compute for inputs
                kwargs = basesum_kwargs if isinstance(node, BaseSum) else dict()
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)
                    else:
                        return node._compute_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("ActualMPEPath"):
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing counts
            self._actual_counts = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)

    def get_mpe_path(self, root, sequence_lens=None):
        if not self._dynamic:
            if sequence_lens is not None:
                raise ValueError(
                    "Sequence length is not none, but this MPEPath instance has not been "
                    "initialized with dynamic=True.")
            self._get_mpe_path_fixed_graph(root)
        else:
            self._get_mpe_path_dynamic_graph(root, sequence_lens=sequence_lens)

    def _get_mpe_path_dynamic_graph(self, root, sequence_lens=None):
        """Assemble TF operations computing the branch counts for the MPE
        downward path through the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        if sequence_lens is not None:
            time0 = root.get_maxlen() - sequence_lens

        def reduce_parents_fun_step(t, node, parent_vals):
            # Sum up all parent vals
            parent_vals = [p for p in parent_vals if p is not None]
            def accumulate():
                if len(parent_vals) > 1:
                    # tf.accumulate_n will complain about a temporary variable being defined more
                    # than once, so use tf.add_n
                    return tf.add_n(parent_vals, name=node.name + "_add")

                return parent_vals[0]

            if node.is_op:
                if sequence_lens is not None:
                    zeros = tf.zeros_like(parent_vals[0])
                    if node.interface_head:
                        return tf.where(tf.less_equal(t, time0), zeros, accumulate())
                    return tf.where(tf.less(t, time0), zeros, accumulate())

                if node.interface_head:
                    # This conditional is needed for dealing with t == 0 or
                    # t <= maxlen - sequence_len.
                    # In that case, the part of the graph under the interface_head
                    # should be disabled (set to zero)
                    return tf.cond(tf.equal(t, 0),
                                   lambda: tf.zeros_like(parent_vals[0]), accumulate)

            # Default behavior
            return accumulate()

        basesum_kwargs = dict(
            add_random=self._add_random, use_unweighted=self._use_unweighted,
            with_ivs=True, sample=self._sample, sample_prob=self._sample_prob,
            sample_rank_based=self._sample_rank_based)
        
        def down_fun_time(t, node, summed):
            if node.is_op:
                # Compute for inputs
                with tf.name_scope(node.name):
                    kwargs = basesum_kwargs if isinstance(node, BaseSum) else dict()
                    if self._log:
                        return node._compute_log_mpe_path(
                            summed, *[self._value.read_node_val(i.node, t)
                                      if i else None for i in node.inputs], **kwargs)
                    else:
                        return node._compute_mpe_path(
                            summed, *[self._value.read_node_val(i.node, t)
                                      if i else None for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root, sequence_lens=sequence_lens)

        with tf.name_scope("MPEPath"):
            # Compute the tensor to feed to the root node
            out_size = root.get_out_size()
            out_size = (out_size,) if isinstance(out_size, int) else out_size
            graph_input_end = tf.ones(shape=(root.get_batch_size(),) + out_size, dtype=conf.dtype)
            graph_input_default = tf.zeros_like(graph_input_end)

            if self._dynamic_reduce_in_loop:
                # Traverse the graph computing counts
                self._true_counts = compute_graph_up_down_dynamic(
                    root, down_fun_step=down_fun_time, graph_input_end=graph_input_end,
                    graph_input_default=graph_input_default,
                    reduce_parents_fun_step=reduce_parents_fun_step, reduce_init=tf.zeros,
                    reduce_binary_op=tf.add)
            else:
                # Traverse the graph computing counts
                self._counts_per_step = compute_graph_up_down_dynamic(
                    root, down_fun_step=down_fun_time, graph_input_end=graph_input_end,
                    graph_input_default=graph_input_default,
                    reduce_parents_fun_step=reduce_parents_fun_step)

                with tf.name_scope("SumAcrossSequence"):
                    for node in self._counts_per_step:
                        self._counts_per_step[node] = per_step = self._counts_per_step[node].stack()
                        self._true_counts[node] = tf.reduce_sum(
                            per_step, axis=0, name=node.name + "CountsTotalPerBatch")
