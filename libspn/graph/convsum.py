# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.inference.type import InferenceType
from libspn.graph.basesum import BaseSum
import libspn.utils as utils
import tensorflow as tf
from libspn.exceptions import StructureError
from libspn.graph.weights import Weights
import numpy as np


@utils.register_serializable
class ConvSum(BaseSum):
    """A container representing multiple par-sums (which share the same input) in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this container.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this container.
        weights (input_like): Input providing weights container to this sum container.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum container. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        name (str): Name of the container.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this container that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, *values, num_channels=1, weights=None, ivs=None,
                 inference_type=InferenceType.MARGINAL, name="ParallelSums",
                 grid_dim_sizes=None):

        if not grid_dim_sizes:
            raise NotImplementedError(
                "{}: Must also provide grid_dim_sizes at this point.".format(self))

        self._grid_dim_sizes = grid_dim_sizes or [-1] * 2
        self._channel_axis = 3
        super().__init__(
            *values, num_sums=num_channels, weights=weights, ivs=ivs,
            inference_type=inference_type, name=name, reduce_axis=4, op_axis=[1, 2])

    def _prepare_component_wise_processing(
            self, w_tensor, ivs_tensor, *input_tensors, zero_prob_val=0.0):
        shape_suffix = [self._num_sums, self._max_sum_size]
        w_tensor = tf.reshape(w_tensor, [1] * 3 + shape_suffix)

        input_tensors = [self._spatial_reshape(t) for t in input_tensors]

        reducible_inputs = utils.concat_maybe(input_tensors, axis=self._reduce_axis)
        if ivs_tensor is not None:
            ivs_tensor = tf.reshape(ivs_tensor, shape=[-1] + self._grid_dim_sizes + shape_suffix)

        return w_tensor, ivs_tensor, reducible_inputs

    def _spatial_reshape(self, t, forward=True):
        non_batch_dim_size = np.prod([ds for i, ds in enumerate(t.shape.as_list())
                                      if i != self._batch_axis])
        input_channels = non_batch_dim_size // np.prod(self._grid_dim_sizes)
        if forward:
            return tf.reshape(t, [-1] + self._grid_dim_sizes + [1, input_channels])
        return tf.reshape(t, [-1] + self._grid_dim_sizes + [self._max_sum_size])

    def _get_sum_sizes(self, num_sums):
        input_sizes = self.get_input_sizes()
        num_values = sum([s // np.prod(self._grid_dim_sizes) 
                          for s in input_sizes[2:]])  # Skip ivs, weights
        return num_sums * np.prod(self._grid_dim_sizes) * [num_values]

    def generate_weights(self, init_value=1, trainable=True, input_sizes=None,
                         log=False, name=None):
        """Generate a weights container matching this sum container and connect it to
        this sum.

        The function calculates the number of weights based on the number
        of input values of this sum. Therefore, weights should be generated
        once all inputs are added to this container.

        Args:
            init_value: Initial value of the weights. For possible values, see
                :meth:`~libspn.utils.broadcast_value`.
            trainable (bool): See :class:`~libspn.Weights`.
            input_sizes (list of int): Pre-computed sizes of each input of
                this container.  If given, this function will not traverse the graph
                to discover the sizes.
            log (bool): If "True", the weights are represented in log space.
            name (str): Name of the weighs container. If ``None`` use the name of the
                        sum + ``_Weights``.

        Return:
            Weights: Generated weights container.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_Weights"
        # Count all input values
        if input_sizes:
            num_values = sum(input_sizes[2:])  # Skip ivs, weights
        else:
            num_values = max(self._sum_sizes)
        # Generate weights
        weights = Weights(
            init_value=init_value, num_weights=num_values, num_sums=self._num_sums,
            log=log, trainable=trainable, name=name)
        self.set_weights(weights)
        return weights
    
    def _compute_out_size(self):
        return int(np.prod(self._grid_dim_sizes) * self._num_sums)

    @utils.lru_cache
    def _compute_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    def _compute_log_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_log_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_mpe_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_log_mpe_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    def _compute_mpe_path_common(
            self, reducible_tensor, counts, w_tensor, ivs_tensor, *input_tensors):
        """Common operations for computing the MPE path.

        Args:
            reducible_tensor (Tensor): A (weighted) ``Tensor`` of (log-)values of this container.
            counts (Tensor): A ``Tensor`` that contains the accumulated counts of the parents
                             of this container.
            w_tensor (Tensor):  A ``Tensor`` containing the (log-)value of the weights.
            ivs_tensor (Tensor): A ``Tensor`` containing the (log-)value of the IVs.
            input_tensors (list): A list of ``Tensor``s with outputs of the child nodes.

        Returns:
            A ``list`` of ``tuple``s [(MPE counts, input tensor), ...] where the first corresponds
            to the Weights of this container, the second corresponds to the IVs and the remaining
            tuples correspond to the nodes in ``self._values``.
        """
        max_indices = self._reduce_argmax(reducible_tensor)
        max_indices = tf.reshape(max_indices, self._compute_out_size())
        max_counts = utils.scatter_values(
            params=counts, indices=max_indices, num_out_cols=self._max_sum_size)
        weight_counts = tf.reduce_sum(max_counts, axis=self._op_axis)
        input_counts = tf.split(tf.reduce_sum(counts, axis=self._channel_axis), self._channel_axis)
        input_counts = [tf.layers.flatten(t) for t in input_counts]
        return self._scatter_to_input_tensors(
            (weight_counts, w_tensor),  # Weights
            (max_counts, ivs_tensor),  # IVs
            *[(t, v) for t, v in zip(input_counts, input_tensors)])  # Values
