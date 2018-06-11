# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode, Input
from libspn.inference.type import InferenceType
from libspn.graph.ivs import IVs
from libspn.graph.weights import Weights
from libspn import utils
from libspn.exceptions import StructureError
from libspn.log import get_logger
from libspn import conf
from libspn.utils.serialization import register_serializable


@register_serializable
class Sums(OpNode):
    """A node representing multiple sums in an SPN, where each sum has it's own
    and seperate input.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this node.
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    logger = get_logger()
    info = logger.info

    def __init__(self, *values, num_sums=1, weights=None, ivs=None,
                 inference_type=InferenceType.MARGINAL, name="Sums"):
        if not num_sums > 0:
            raise StructureError("In %s num_sums: %s need to be > 0" % self, num_sums)

        super().__init__(inference_type, name)

        self.set_values(*values)
        self._num_sums = num_sums
        self.set_weights(weights)
        self.set_ivs(ivs)

# TODO
    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        data['num_sums'] = self._num_sums
        if self._weights:
            data['weights'] = (self._weights.node.name, self._weights.indices)
        if self._ivs:
            data['ivs'] = (self._ivs.node.name, self._ivs.indices)
        return data

# TODO
    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()
        self._num_sums = data['num_sums']
        self.set_weights()
        self.set_ivs()

# TODO
    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])
        weights = data.get('weights', None)
        if weights:
            self._weights = Input(nodes_by_name[weights[0]], weights[1])
        ivs = data.get('ivs', None)
        if ivs:
            self._ivs = Input(nodes_by_name[ivs[0]], ivs[1])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return (self._weights, self._ivs) + self._values

    @property
    def num_sums(self):
        """int: Number of Sum ops modelled by this node."""
        return self._num_sums

    def set_num_sums(self, num_sums=1):
        """Set the number of Sum ops modelled by this node.

        Args:
            num_sums (int): Number of Sum ops modelled by this node.
        """
        self._num_sums = num_sums

    @property
    def weights(self):
        """Input: Weights input."""
        return self._weights

    def set_weights(self, weights=None):
        """Set the weights input.

        Args:
            weights (input_like): Input providing weights node to this sum node.
                See :meth:`~libspn.Input.as_input` for possible values. If set
                to ``None``, the input is disconnected.
        """
        weights, = self._parse_inputs(weights)
        if weights and not isinstance(weights.node, Weights):
            raise StructureError("%s is not Weights" % weights.node)
        self._weights = weights

    @property
    def ivs(self):
        """Input: IVs input."""
        return self._ivs

    def set_ivs(self, ivs=None):
        """Set the IVs input.

        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        """
        self._ivs, = self._parse_inputs(ivs)

    @property
    def values(self):
        """list of Input: List of value inputs."""
        return self._values

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._parse_inputs(*values)

    def add_values(self, *values):
        """Add more inputs providing input values to this node.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._values + self._parse_inputs(*values)

    def generate_weights(self, init_value=1, trainable=True,
                         input_sizes=None, name=None):
        """Generate a weights node matching this sum node and connect it to
        this sum.

        The function calculates the number of weights based on the number
        of input values of this sum. Therefore, weights should be generated
        once all inputs are added to this node.

        Args:
            init_value: Initial value of the weights. For possible values, see
                :meth:`~libspn.utils.broadcast_value`.
            trainable (bool): See :class:`~libspn.Weights`.
            input_sizes (list of int): Pre-computed sizes of each input of
                this node.  If given, this function will not traverse the graph
                to discover the sizes.
            name (str): Name of the weighs node. If ``None`` use the name of the
                        sum + ``_Weights``.

        Return:
            Weights: Generated weights node.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_Weights"
        # Count all input values
        if not input_sizes:
            input_sizes = self.get_input_sizes()
        num_values = sum(input_sizes[2:])  # Skip ivs, weights

        if init_value == 1:
            init_value = utils.broadcast_value(1,
                                               (num_values,),
                                               dtype=conf.dtype)

        # Generate weights
        weights = Weights(init_value=init_value,
                          num_weights=int(num_values / self._num_sums),
                          num_sums=self._num_sums,
                          trainable=trainable, name=name)
        self.set_weights(weights)
        return weights

    def generate_ivs(self, feed=None, name=None):
        """Generate an IVs node matching this sum node and connect it to
        this sum.

        IVs should be generated once all inputs are added to this node,
        otherwise the number of IVs will be incorrect.

        Args:
            feed (Tensor): See :class:`~libspn.IVs`.
            name (str): Name of the IVs node. If ``None`` use the name of the
                        sum + ``_IVs``.

        Return:
            IVs: Generated IVs node.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_IVs"
        # Count all input values
        num_values = sum(len(v.indices) if v.indices is not None
                         else v.node.get_out_size()
                         for v in self._values)
        ivs = IVs(feed=feed, num_vars=self._num_sums,
                  num_vals=int(num_values / self._num_sums),
                  name=name)
        self.set_ivs(ivs)
        return ivs

    @property
    def _const_out_size(self):
        return True

    def _compute_out_size(self, *input_out_sizes):
        return self._num_sums

    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, ivs_scopes, *value_scopes = self._gather_input_scopes(weight_scopes,
                                                                 ivs_scopes,
                                                                 *value_scopes)
        flat_value_scopes = list(chain.from_iterable(value_scopes))
        sublist_size = int(len(flat_value_scopes) / self._num_sums)
        # Divide gathered value scopes into sublists, one per modelled Sum node
        value_scopes_sublists = [flat_value_scopes[i:i+sublist_size] for i in
                                 range(0, len(flat_value_scopes), sublist_size)]
        if self._ivs:
            sublist_size = int(len(ivs_scopes) / self._num_sums)
            # Divide gathered ivs scopes into sublists, one per modelled Sum node
            ivs_scopes_sublists = [ivs_scopes[i:i+sublist_size] for i in
                                   range(0, len(ivs_scopes), sublist_size)]
            # Add respective ivs scope to value scope list of each Sum node
            for val, ivs in zip(value_scopes_sublists, ivs_scopes_sublists):
                val.extend(ivs)
        return [Scope.merge_scopes(val_scope) for val_scope in
                value_scopes_sublists]

    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, ivs_scopes_, *value_scopes_ = self._gather_input_scopes(weight_scopes,
                                                                   ivs_scopes,
                                                                   *value_scopes)
        # If already invalid, return None
        if (any(s is None for s in value_scopes_)
                or (self._ivs and ivs_scopes_ is None)):
            return None
        flat_value_scopes = list(chain.from_iterable(value_scopes_))
        # IVs
        if self._ivs:
            # Verify number of IVs
            if len(ivs_scopes_) != len(flat_value_scopes):
                raise StructureError("Number of IVs (%s) and values (%s) does "
                                     "not match for %s"
                                     % (len(ivs_scopes_), len(flat_value_scopes),
                                        self))
            # Check if scope of all IVs is just one and the same variable
            if len(Scope.merge_scopes(ivs_scopes_)) > self._num_sums:
                return None
        # Check sum for completeness wrt values
        first_scope = flat_value_scopes[0]
        if any(s != first_scope for s in flat_value_scopes[1:]):
            Sums.info("%s is not complete with input value scopes %s",
                      self, flat_value_scopes)
            return None
        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes)

    def _compute_value_common(self, weight_tensor, ivs_tensor, *value_tensors):
        """Common actions when computing value."""
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if not self._weights:
            raise StructureError("%s is missing weights" % self)
        # Prepare values
        weight_tensor, ivs_tensor, *value_tensors = self._gather_input_tensors(
         weight_tensor, ivs_tensor, *value_tensors)
        values = utils.concat_maybe(value_tensors, 1)
        return weight_tensor, ivs_tensor, values

    def _compute_value(self, weight_tensor, ivs_tensor, *value_tensors):
        weight_tensor, ivs_tensor, values = self._compute_value_common(
            weight_tensor, ivs_tensor, *value_tensors)
        if self._ivs:
            values_selected = values * ivs_tensor
        else:
            values_selected = values
        # Shape of values_selected = (Batch X (num_sums * num_vals))
        # reshape it to (Batch X num_sums X num_feat)
        reshape = (-1, self._num_sums, int(values.shape[1].value / self._num_sums))
        values_reshaped = tf.reshape(values_selected, shape=reshape)
        values_selected_weighted = values_reshaped * weight_tensor
        return tf.reduce_sum(values_selected_weighted, axis=2)

    def _compute_log_value(self, weight_tensor, ivs_tensor, *value_tensors):
        weight_tensor, ivs_tensor, values = self._compute_value_common(
            weight_tensor, ivs_tensor, *value_tensors)
        if self._ivs:
            values_selected = values + ivs_tensor
        else:
            values_selected = values
        # Shape of values_selected = (Batch X (num_sums * num_vals))
        # reshape it to (Batch X num_sums X num_feat)
        reshape = (-1, self._num_sums, int(values.shape[1].value / self._num_sums))
        values_reshaped = tf.reshape(values_selected, shape=reshape)
        values_selected_weighted = values_reshaped + weight_tensor
        return utils.reduce_log_sum_3D(values_selected_weighted, transpose=False)

    def _compute_mpe_value(self, weight_tensor, ivs_tensor, *value_tensors):
        weight_tensor, ivs_tensor, values = self._compute_value_common(
            weight_tensor, ivs_tensor, *value_tensors)
        if self._ivs:
            values_selected = values * ivs_tensor
        else:
            values_selected = values
        # Shape of values_selected = (Batch X (num_sums * num_vals))
        # reshape it to (Batch X num_sums X num_feat)
        reshape = (-1, self._num_sums, int(values.shape[1].value / self._num_sums))
        values_reshaped = tf.reshape(values_selected, shape=reshape)
        values_selected_weighted = values_reshaped * weight_tensor
        return tf.reduce_max(values_selected_weighted, axis=2)

    def _compute_log_mpe_value(self, weight_tensor, ivs_tensor, *value_tensors):
        weight_tensor, ivs_tensor, values = self._compute_value_common(
            weight_tensor, ivs_tensor, *value_tensors)
        if self._ivs:
            values_selected = values + ivs_tensor
        else:
            values_selected = values
        # Shape of values_selected = (Batch X (num_sums * num_vals))
        # reshape it to (Batch X num_sums X num_feat)
        reshape = (-1, self._num_sums, int(values.shape[1].value / self._num_sums))
        values_reshaped = tf.reshape(values_selected, shape=reshape)
        values_selected_weighted = values_reshaped + weight_tensor
        return tf.reduce_max(values_selected_weighted, axis=2)

    def _compute_mpe_path_common(self, values_weighted, counts, weight_value,
                                 ivs_value, *value_values):
        # Propagate the counts to the max value
        max_indices = tf.argmax(values_weighted, axis=2)
        max_counts = utils.scatter_values(params=counts, indices=max_indices,
                                          num_out_cols=values_weighted.shape[2].value)
        _, _, *value_sizes = self.get_input_sizes(None, None, *value_values)
        # Reshape max counts to a wide 2D tensor of shape 'Batch X (num_sums * num_vals)'
        reshape = (-1, sum(value_sizes))
        max_counts_reshaped = tf.reshape(max_counts, shape=reshape)
        # Split the reshaped max counts to value inputs
        max_counts_split = tf.split(max_counts_reshaped, value_sizes, 1)
        return self._scatter_to_input_tensors(
            (max_counts, weight_value),  # Weights
            (max_counts_reshaped, ivs_value),  # IVs
            *[(t, v) for t, v in zip(max_counts_split, value_values)])  # Values

    def _compute_mpe_path(self, counts, weight_value, ivs_value, *value_values,
                          add_random=None, use_unweighted=False):
        # Get weighted, IV selected values
        weight_tensor, ivs_tensor, values = self._compute_value_common(
            weight_value, ivs_value, *value_values)
        if self._ivs:
            values_selected = values * ivs_tensor
        else:
            values_selected = values
        # Shape of values_selected = (Batch X (num_sums * num_vals))
        # reshape it to (Batch X num_sums X num_feat)
        reshape = (-1, self._num_sums, int(values.shape[1].value / self._num_sums))
        values_reshaped = tf.reshape(values_selected, shape=reshape)
        values_selected_weighted = values_reshaped * weight_tensor
        return self._compute_mpe_path_common(values_selected_weighted, counts,
                                             weight_value, ivs_value, *value_values)

    def _compute_log_mpe_path(self, counts, weight_value, ivs_value,
                              *value_values, add_random=None,
                              use_unweighted=False):
        # Get weighted, IV selected values
        weight_value, ivs_value, values = self._compute_value_common(
            weight_value, ivs_value, *value_values)
        if self._ivs:
            values_selected = values + ivs_value
        else:
            values_selected = values
        # Shape of values_selected = (Batch X (num_sums * num_vals))
        # reshape it to (Batch X num_sums X num_feat)
        reshape = (-1, self._num_sums, int(values.shape[1].value / self._num_sums))
        values_reshaped = tf.reshape(values_selected, shape=reshape)

        # WARN USING UNWEIGHTED VALUE
        if not use_unweighted or any(v.node.is_var for v in self._values):
            values_weighted = values_reshaped + weight_value
        else:
            values_weighted = values_reshaped
        # / USING UNWEIGHTED VALUE

        # WARN ADDING RANDOM NUMBERS
        if add_random is not None:
            values_weighted = tf.add(values_weighted, tf.random_uniform(
                shape=(tf.shape(values_weighted)[0], 1,
                       values_weighted.shape[2].value),
                minval=0, maxval=add_random,
                dtype=conf.dtype))
        # /ADDING RANDOM NUMBERS

        return self._compute_mpe_path_common(
            values_weighted, counts, weight_value, ivs_value, *value_values)
