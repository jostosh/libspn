# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain, combinations
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode, Input
from libspn.inference.type import InferenceType
from libspn import utils
from libspn.exceptions import StructureError
from libspn.log import get_logger
from libspn.utils.serialization import register_serializable


@register_serializable
class Product(OpNode):
    """A node representing a single product in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        name (str): Name of the node.
    """

    __logger = get_logger()
    __info = __logger.info

    def __init__(self, *values, name="Product"):
        self._values = []
        super().__init__(InferenceType.MARGINAL, name)
        self.set_values(*values)

    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()

    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return self._values

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
        return 1

    def _compute_scope(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes = self._gather_input_scopes(*value_scopes)
        return [Scope.merge_scopes(chain.from_iterable(value_scopes))]

    def _compute_valid(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes_ = self._gather_input_scopes(*value_scopes)
        # If already invalid, return None
        if any(s is None for s in value_scopes_):
            return None
        # Check product decomposability
        flat_value_scopes = list(chain.from_iterable(value_scopes_))
        for s1, s2 in combinations(flat_value_scopes, 2):
            if s1 & s2:
                self.__info("%s is not decomposable with input value scopes %s",
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
        return values

    def _compute_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        return tf.reduce_prod(values, 1, keep_dims=True)

    def _compute_log_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        return tf.reduce_sum(values, 1, keep_dims=True)

    def _compute_mpe_value(self, *value_tensors):
        return self._compute_value(*value_tensors)

    def _compute_log_mpe_value(self, *value_tensors):
        return self._compute_log_value(*value_tensors)

    def _compute_mpe_path(self, counts, *value_values, add_random=False,
                          use_unweighted=False, with_ivs=False):
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)

        def process_input(v_input, v_value):
            input_size = v_input.get_size(v_value)
            # Tile the counts if input is larger than 1
            return (tf.tile(counts, [1, input_size])
                    if input_size > 1 else counts)

        # For each input, pass counts to all elements selected by indices
        value_counts = [(process_input(v_input, v_value), v_value)
                        for v_input, v_value
                        in zip(self._values, value_values)]
        # TODO: Scatter to input tensors can be merged with tiling to reduce
        # the amount of operations.
        return self._scatter_to_input_tensors(*value_counts)

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False,
                              use_unweighted=False, with_ivs=False):
        return self._compute_mpe_path(counts, *value_values)
