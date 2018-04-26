# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain, combinations, repeat
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode, Input
from libspn.inference.type import InferenceType
from libspn import utils
from libspn.exceptions import StructureError
from libspn.log import get_logger
from libspn.utils.serialization import register_serializable


@register_serializable
class Products(OpNode):
    """A node representing a multiple products in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_prods (int): Number of Product ops modelled by this node.
        name (str): Name of the node.
    """

    logger = get_logger()
    info = logger.info

    def __init__(self, *values, num_prods=1, name="Products"):
        if not num_prods > 0:
            raise StructureError("In %s num_prods: %s need to be > 0" % self, num_prods)

        self._values = []
        self._num_prods = num_prods
        super().__init__(InferenceType.MARGINAL, name)
        self.set_values(*values)

    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        data['num_prods'] = self._num_prods
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()
        self._num_prods = data['num_prods']

    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return self._values

    @property
    def num_prods(self):
        """int: Number of Product ops modelled by this node."""
        return self._num_prods

    def set_num_prods(self, num_prods=1):
        """Set the number of Product ops modelled by this node.

        Args:
            num_prods (int): Number of Product ops modelled by this node.
        """
        self._num_prods = num_prods

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

    @property
    def _const_out_size(self):
        return True

    def _compute_out_size(self, *input_out_sizes):
        return self._num_prods

    def _compute_scope(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes = list(chain.from_iterable(self._gather_input_scopes(
                                                *value_scopes)))
        sublist_size = int(len(value_scopes) / self._num_prods)
        # Divide gathered value scopes into sublists, one per modelled Product node.
        value_scopes_sublists = [value_scopes[i:i+sublist_size] for i in
                                 range(0, len(value_scopes), sublist_size)]
        return [Scope.merge_scopes(vs) for vs in value_scopes_sublists]

    def _compute_valid(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes_ = self._gather_input_scopes(*value_scopes)
        # If already invalid, return None
        if any(s is None for s in value_scopes_):
            return None
        # Check product decomposability
        flat_value_scopes = list(chain.from_iterable(value_scopes_))
        values_per_product = int(len(flat_value_scopes) / self._num_prods)
        sub_value_scopes = [flat_value_scopes[i:(i + values_per_product)] for i in
                            range(0, len(flat_value_scopes), values_per_product)]
        for scopes in sub_value_scopes:
            for s1, s2 in combinations(scopes, 2):
                if s1 & s2:
                    Products.info("%s is not decomposable with input value scopes %s",
                                  self, flat_value_scopes)
                    return None
        return self._compute_scope(*value_scopes)

    def _compute_value_common(self, *value_tensors):
        """Common actions when computing value."""
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        # Prepare values
        value_tensors = self._gather_input_tensors(*value_tensors)
        if len(value_tensors) > 1:
            values = tf.concat(values=value_tensors, axis=1)
        else:
            values = value_tensors[0]
        if self._num_prods > 1:
            # Shape of values tensor = [Batch, (num_prods * num_vals)]
            # First, split the values tensor into 'num_prods' smaller tensors.
            # Then pack the split tensors together such that the new shape
            # of values tensor = [Batch, num_prods, num_vals]
            reshape = (-1, self._num_prods, int(values.shape[1].value /
                                                self._num_prods))
            reshaped_values = tf.reshape(values, shape=reshape)
            return reshaped_values
        else:
            return values

    def _compute_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        return tf.reduce_prod(values, axis=-1, keep_dims=(False if
                              self._num_prods > 1 else True))

    def _compute_log_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        return tf.reduce_sum(values, axis=-1,
                             keep_dims=(False if self._num_prods > 1 else True))

    def _compute_mpe_value(self, *value_tensors):
        return self._compute_value(*value_tensors)

    def _compute_log_mpe_value(self, *value_tensors):
        return self._compute_log_value(*value_tensors)

    def _compute_mpe_path(self, counts, *value_values, add_random=False, use_unweighted=False):
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)

        value_sizes = self.get_input_sizes(*value_values)
        input_size_per_prod = sum(value_sizes) // self._num_prods

        # (1-3) Tile counts of each prod based on prod-input-size, by gathering
        indices = list(chain.from_iterable(repeat(r, input_size_per_prod)
                       for r in range(self._num_prods)))
        gathered_counts = utils.gather_cols(counts, indices)

        # (4) Split gathered countes based on value_sizes
        value_counts = tf.split(gathered_counts, value_sizes, axis=1)
        counts_values_paired = [(v_count, v_value) for v_count, v_value in
                                zip(value_counts, value_values)]

        # (5) scatter_cols (num_inputs)
        return self._scatter_to_input_tensors(*counts_values_paired)

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False, use_unweighted=False):
        return self._compute_mpe_path(counts, *value_values)
