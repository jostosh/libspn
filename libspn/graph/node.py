# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
import abc
from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple, OrderedDict

import tensorflow as tf
from libspn import utils, conf
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
from libspn.graph.algorithms import compute_graph_up, traverse_graph


class GraphData():
    """Data structure holding information common for all SPN nodes
    in the same TensorFlow graph.

    Right now, we do not keep here anything, but we were, and we might in
    the future, therefore we keep this class for now.
    """

    def __init__(self, tf_graph):
        self._tf_graph = tf_graph

    @property
    def tf_graph(self):
        return self._tf_graph

    @staticmethod
    def get(tf_graph=None):
        """Get GraphData for the TF graph or install a new GraphData if the
        TF graph does not have one yet.

        Args:
            graph: The TF graph to use, if ``None``, use current.

        Return:
            GraphData: Graph data attached to the TF graph.
        """
        if tf_graph is None:
            tf_graph = tf.get_default_graph()
        if not hasattr(tf_graph, 'spn_data'):
            tf_graph.spn_data = GraphData(tf_graph)
        return tf_graph.spn_data


class Input():

    """Holds information about a single input of an operation node. The input
    can be either disconnected (in which case it casts to ``False``) or
    connected to selected elements of an output of a specific node. The
    elements to which the input is connected (and their order) are indicated
    using ``indices``. If indices is set to ``None`` all output elements are
    used in the order in which they are produced.

    Attributes:
        node (Node): The node attached to the input. If set to ``None``, this
                     input is disconnected and has nothing attached.
        indices (int or list of int): A list of indices of elements in the
            tensor produced by the input node that will be attached to the
            input. The indices do not have to be sorted and their order will
            specify the order in which the elements are attached to the input.
            If ``indices`` is ``None``, all elements are attached to the input
            in the order in which they appear in the tensor produced by the
            input node.
    """

    def __init__(self, node=None, indices=None):
        # Disconnected input?
        if node is None:
            self.node = None
            self.indices = None
            return

        # Verify node
        if not isinstance(node, Node):
            raise TypeError("Input node %s is not a Node" % (node,))
        self.node = node

        # Wrap indices in a list
        if isinstance(indices, int):
            indices = [indices]

        # Verify indices
        if isinstance(indices, list):
            # List empty?
            if not indices:
                raise ValueError("Indices for node %s are an empty list"
                                 % (node,))
            # Verify index values
            if any(not isinstance(j, int) or j < 0 for j in indices):
                raise ValueError("Indices %s for node %s are not non-negative"
                                 " integers" % (indices, node))
            # Check for duplicates - duplicated indices cannot be handled
            # properly during the downward pass since integrating multiple
            # parents happens only on the level of inputs, not indices.
            if len(set(indices)) != len(indices):
                raise ValueError("Indices %s for node %s contain duplicates"
                                 % (indices, node))
        elif indices is not None:
            raise TypeError("Invalid indices %s for node %s" % (indices, node))
        self.indices = indices

    @classmethod
    def as_input(cls, value):
        """Convert ``value`` to a valid :class:`Input` if it is not
        :class:`Input` already.

        Args:
            value: An :class:`Input`, a :class:`Node`, or a tuple
                   ``(node, indices)``, where indices can be a single index,
                   a list of integer indices or ``None``.
        """
        # None
        if value is None:
            return cls(None)
        # Input
        if isinstance(value, cls):
            return value
        # Node
        elif isinstance(value, Node):
            return cls(value, None)
        # Tuple
        elif isinstance(value, tuple) and len(value) == 2:
            return cls(value[0], value[1])
        else:
            raise TypeError("Cannot convert %s to an input" % (value,))

    @property
    def is_op(self):
        """Returns ``True`` if the input is connected to an operation node."""
        return isinstance(self.node, OpNode)

    @property
    def is_param(self):
        """Returns ``True`` if the input is connected to a parameter node."""
        return isinstance(self.node, ParamNode)

    @property
    def is_var(self):
        """Returns ``True`` if the input is connected to a variable node."""
        return isinstance(self.node, VarNode)

    @property
    def is_distribution(self):
        return isinstance(self.node, DistributionNode)

    def get_size(self, input_tensor):
        """Get the size of the input.

        Args:
            input_tensor (Tensor): The tensor produced by the node connected
                                   to the input.

        Return:
           int: Size of the input.
        """
        # Get input node output size from tensor
        input_tensor_shape = input_tensor.get_shape()
        if input_tensor_shape.ndims == 1:
            out_size = int(input_tensor_shape[0])
        else:
            out_size = int(input_tensor_shape[1])
        # Calculate input size
        return out_size if self.indices is None else len(self.indices)

    def __bool__(self):
        """Returns True if a node is connected to this input."""
        return self.node is not None

    def __repr__(self):
        return "Input(%s, %s)" % (self.node, self.indices)

    def __eq__(self, other):
        if not isinstance(other, Input):
            raise ValueError("Cannot compare Input and %s" %
                             type(other))
        return self.node is other.node and self.indices == other.indices

    def __ne__(self, other):
        if not isinstance(other, Input):
            raise ValueError("Cannot compare Input and %s" %
                             type(other))
        return self.node is not other.node or self.indices != other.indices


class Node(ABC):
    """An abstract class defining the interface of a node of the SPN graph.

    Args:
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, inference_type, name):
        self._graph_data = GraphData.get()
        if name is None:
            name = "Node"
        self._name = self.tf_graph.unique_name(name)
        self.inference_type = inference_type
        with tf.name_scope(self._name + "/"):
            self._create()

    @abstractmethod
    def serialize(self):
        """Convert the data in this node into a dictionary for serialization.

        Returns:
            dict: Dictionary with all the data to be serialized.
        """
        return {'name': self._name,
                'inference_type': self.inference_type.name}

    @abstractmethod
    def deserialize(self, data):
        """Initialize this node with the ``data`` dict during deserialization.

        Args:
            data (dict): Dictionary with all the data to be deserialized.
        """
        Node.__init__(self, name=data['name'],
                      inference_type=InferenceType[data['inference_type']])

    @property
    def name(self):
        """str: Name of the node."""
        return self._name

    @property
    def tf_graph(self):
        """TensorFlow graph with which this SPN graph node is associated."""
        return self._graph_data.tf_graph

    @property
    def is_op(self):
        """Returns ``True`` if the node is an operation node."""
        # Not the best oop, but avoids the need for importing .node to check
        return isinstance(self, OpNode)

    @property
    def is_param(self):
        """Returns ``True`` if the node is a parameter node."""
        # Not the best oop, but avoids the need for importing .node to check
        return isinstance(self, ParamNode)

    @property
    def is_var(self):
        """Returns ``True`` if the node is a variable node."""
        # Not the best oop, but avoids the need for importing .node to check
        return isinstance(self, VarNode)

    def get_tf_graph_size(self):
        """Get the size of the TensorFlow graph with which this SPN graph node is associated."""
        return len(self.tf_graph.get_operations())

    def get_nodes(self, skip_params=False):
        """Get a list of nodes in the (sub-)graph rooted in this node.

        Args:
            skip_params (bool): If ``True``, param nodes will not be included.

        Returns:
            list of Node: List of nodes.
        """
        nodes = []
        traverse_graph(self, fun=lambda node: nodes.append(node),
                       skip_params=skip_params)
        return nodes

    def get_num_nodes(self, skip_params=False):
        """Get the number of nodes in the SPN graph for which this node is root.

        Args:
            skip_params (bool): If ``True`` don't count param nodes.

        Returns:
            int: Number of nodes.
        """
        class Counter:
            """"Mutable int."""

            def __init__(self):
                self.val = 0

            def inc(self):
                self.val += 1

        c = Counter()
        traverse_graph(self, fun=lambda node: c.inc(),
                       skip_params=skip_params)
        return c.val

    def get_out_size(self):
        """Get the size of the output of this node.  The size might depend on
        the inputs of this node and might change if new inputs are added.

        Returns:
            int: The size of the output.
        """
        return compute_graph_up(self,
                                (lambda node, *args:
                                 node._compute_out_size(*args)),
                                (lambda node: node._const_out_size))

    def get_scope(self):
        """Get the scope of each output value of this node.

        Returns:
            list of Scope: A list of length ``out_size`` containing scopes of
                           each output of this node.
        """
        return compute_graph_up(self, (lambda node, *args:
                                       node._compute_scope(*args)))

    def is_valid(self):
        """Check if the SPN rooted in this node is complete and decomposable.
        If a node has multiple outputs, it is considered valid if all outputs
        of that node come from a valid SPN.

        Returns:
            bool: ``True`` if this SPN is complete and decomposable.
        """
        return (compute_graph_up(self, (lambda node, *args:
                                        node._compute_valid(*args)))
                is not None)

    def get_value(self, inference_type=None):
        """Assemble TF operations computing the value of the SPN rooted in
        this node.

        Args:
            inference_type (InferenceType): Determines the type of inference
                that should be used. If set to ``None``, the inference type is
                specified by the ``inference_type`` flag of the node. If set to
                ``MARGINAL``, marginal inference will be used for all nodes. If
                set to ``MPE``, MPE inference will be used for all nodes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """
        from libspn.inference.value import Value
        return Value(inference_type).get_value(self)

    def get_log_value(self, inference_type=None):
        """Assemble TF operations computing the log value of the SPN rooted in
        this node.

        Args:
            inference_type (InferenceType): Determines the type of inference
                that should be used. If set to ``None``, the inference type is
                specified by the ``inference_type`` flag of the node. If set to
                ``MARGINAL``, marginal inference will be used for all nodes. If
                set to ``MPE``, MPE inference will be used for all nodes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """
        from libspn.inference.value import LogValue
        return LogValue(inference_type).get_value(self)

    def set_inference_types(self, inference_type):
        """Set inference type for each node in the SPN rooted in this node.

        Args:
           inference_type (InferenceType): Inference type to set for the nodes.
        """
        def fun(node):
            node.inference_type = inference_type

        traverse_graph(self, fun=fun, skip_params=False)

    def _create(self):
        """Create any TF placeholder or variable that need to be instantiated
        during the creation of the node and shared between all operations.

        To be re-implemented in a subclass.
        """

    @abstractproperty
    def _const_out_size(self):
        """bool: If True, the number of outputs of this node does not depend
        on the inputs of the node and is fixed.

        To be re-implemented in sub-classes.
        """

    @abstractmethod
    def _compute_out_size(self, *input_out_sizes):
        """Compute the size of the output of this node.

        To be re-implemented in sub-classes.

        Args:
            *input_out_sizes (int): For each input, the size of the output of
                                    the input node.

        Returns:
            int: Size of the output of this node.
        """

    @abstractmethod
    def _compute_scope(self, *input_scopes):
        """Compute the scope of each output value of this node.

        To be re-implemented in sub-classes.

        Args:
            *input_scopes (list of Scope): For each input, scopes of all output
                                           values of the input node.

        Returns:
            list of Scope: A list of length ``out_size`` containing scopes of
            all output values of this node.
        """

    @abstractmethod
    def _compute_valid(self, *input_scopes):
        """Check for validity of the SPN rooted in this node. If the node has
        multiple outputs, it is considered valid if all outputs of that node
        come from a valid SPN.

        If valid, return the scope of each output value of this node, otherwise,
        return ``None`` to indicate that the node/SPN is not valid.

        To be re-implemented in sub-classes.

        Args:
            *input_scopes (list of Scope): For each input, scopes of all output
                 values of the input node or ``None`` if the SPN was found to be
                 invalid already.

        Returns:
            list of Scope: A list of length ``out_size`` containing scopes of
            all output of this node if the SPN rooted in this node is valid,
            otherwise ``None``.
        """

    @abstractmethod
    def _compute_value(self, *input_tensors):
        """Assemble TF operations computing the marginal value of this node.

        To be re-implemented in sub-classes.

        Args:
            *input_tensors (Tensor): For each input, a tensor produced by
                                     the input node.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """

    @abstractmethod
    def _compute_log_value(self, *input_tensors):
        """Assemble TF operations computing the marginal log value of this node.

        To be re-implemented in sub-classes.

        Args:
            *input_tensors (Tensor): For each input, a tensor produced by
                                     the input node.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """

    @abstractmethod
    def _compute_mpe_value(self, *input_tensors):
        """Assemble TF operations computing the MPE value of this node.

        To be re-implemented in sub-classes.

        Args:
            *input_tensors (Tensor): For each input, a tensor produced by
                                     the input node.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """

    @abstractmethod
    def _compute_log_mpe_value(self, *input_tensors):
        """Assemble TF operations computing the log MPE value of this node.

        To be re-implemented in sub-classes.

        Args:
            *input_tensors (Tensor): For each input, a tensor produced by
                                     the input node.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """

    def __repr__(self):
        return self._name

    def __ge__(self, other):
        """Enables sorting nodes."""
        return id(self) >= id(other)

    def __gt__(self, other):
        """Enables sorting nodes."""
        return id(self) > id(other)

    def __le__(self, other):
        """Enables sorting nodes."""
        return id(self) <= id(other)

    def __lt__(self, other):
        """Enables sorting nodes."""
        return id(self) < id(other)


class OpNode(Node):
    """An abstract class defining an operation node of the SPN graph.

    Args:
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, inference_type=InferenceType.MARGINAL,
                 name=None):
        super().__init__(inference_type, name)

    @abstractmethod
    def deserialize_inputs(self, data, nodes_by_name):
        """Attach inputs to this node during deserialization.

        Args:
            data (dict): Dictionary with all the data to be deserialized.
            nodes_by_name (dict): Dictionary of nodes indexed by their original
                                  name.
        """

    @abstractproperty
    def inputs(self):
        """list of Input: Inputs of this node."""
        return tuple()

    def get_input_sizes(self, *input_tensors):
        """Get the sizes of inputs of this node (as selected by indices).
        If the input is disconnected, ``None`` is returned for that input.

        Args:
            *input_tensors (Tensor): Optional tensors with values produced by
                the nodes connected to the inputs. If not given, the input sizes
                will be computed by traversing the graph. If given, the input
                sizes will be computed based on the sizes of ``input_tensors``.
                If ``None`` is given for an input, ``None`` is returned for that
                input.

        Returns:
            list of int: For each input, the size of the input.
        """
        def val_fun(node, *args):
            if node is self:
                return self._gather_input_sizes(*args)
            else:
                return node._compute_out_size(*args)

        def const_fun(node):
            if node is self:
                # Make sure to go through the children of this node
                return False
            else:
                return node._const_out_size

        if input_tensors:
            if len(self.inputs) != len(input_tensors):
                raise ValueError("Number of 'input_tensors' must be the same"
                                 " as the number of inputs.")
            return tuple(None if not inpt or tensor is None
                         else inpt.get_size(tensor)
                         for inpt, tensor
                         in zip(self.inputs, input_tensors))
        else:
            return compute_graph_up(self, val_fun=val_fun, const_fun=const_fun)

    def _parse_inputs(self, *input_likes):
        """Convert the given input_like values to Inputs and verify that the
        inputs are compatible with this node.

        Args:
            *input_likes (input_like): Input descriptions. See
                :meth:`~libspn.Input.as_input` for possible values.

        Returns:
            tuple of Input: Tuple of :class:``~libspn.Input``, one for each
            argument.
        """
        def convert(input_like):
            inpt = Input.as_input(input_like)
            if inpt and inpt.node.tf_graph is not self.tf_graph:
                raise StructureError("%s is in a different TF graph than %s"
                                     % (inpt.node, self))
            return inpt

        return tuple(convert(i) for i in input_likes)

    def _gather_input_sizes(self, *input_out_sizes):
        """For each input, count the input values selected by the input indices.
        If the input is disconnected or ``None`` is given as input_out_size,
        ``None`` is returned for that input.

        Args:
            *input_out_sizes (int): For each input, the size of the output of
                                    the input node.

        Returns:
            list of int: For each input, the size of the input.
        """
        return tuple(None if not inpt or s is None
                     else s if inpt.indices is None
                     else len(inpt.indices)
                     for inpt, s in zip(self.inputs, input_out_sizes))

    def _gather_input_scopes(self, *input_scopes):
        """For each input, gather the scopes of input node output values
        selected by the input indices. If the input is disconnected or ``None``
        is given as input scopes for the input, ``None`` is returned for that
        input.

        Args:
            *input_scopes (list of Scope): For each input, scopes of all output
                                           values of the input node.

        Returns:
            tuple of list of Scopes: For each input, scopes of the output
            values of the input node which are selected by indices (and in the
            order indicated by indices) or ``None``.
        """
        return tuple(None if not inpt or s is None
                     else s if inpt.indices is None
                     else [s[index] for index in inpt.indices]
                     for (inpt, s) in zip(self.inputs, input_scopes))

    def _gather_input_tensors(self, *input_tensors):
        """For each input, gather the elements of the tensor output by the
        input node. The elements indicated by the input indices are gathered
        in the order given by the input indices into a single tensor. If input
        indices are ``None``, it adds no operations for the input tensor and
        forwards it as is. If the input is disconnected or ``None`` is given as
        the input tensor, ``None`` is returned for that input.

        Args:
            *inputs_tensors (Tensor): For each input, a tensor produced by the
                                      input node.

        Returns:
            list of Tensor: For each input, a tensor of shape ``[None, num_elems]``,
            where the first dimension corresponds to the batch size, and
            ``num_elems`` is the number of elements of the input tensor selected
            by the input indices.
        """
        with tf.name_scope("gather_input_tensors", values=input_tensors):
            return tuple(None if not i or it is None
                         else it if i.indices is None
                         else utils.gather_cols(it, i.indices)
                         for i, it in
                         zip(self.inputs, input_tensors))

    def _scatter_to_input_tensors(self, *tuples):
        """For each input, scatter the given tensor to elements indicated by
        input indices. This reverses what ``gather_input_tensors`` is doing.
        If input indices are ``None``, it adds no operations and forwards the
        tensor as is. If the input is disconnected or ``None`` is given in
        ``*tuples``, ``None`` is returned for that input.

        Args:
            *tuples (tuple): For each input, a tuple ``(tensor, input_tensor)``,
                where ``tensor`` is the tensor to be scattered, and
                ``input_tensor`` is the tensor produced by the input node. The
                second tensor is used only to retrieve the appropriate dimensions.

        Returns:
            list of Tensor: A list of tensors containing scattered values.
        """
        with tf.name_scope("scatter_to_input_tensors", values=[t[0] for t in tuples]):
            return tuple(None if not i or t is None
                         else t[0] if i.indices is None
                         else utils.scatter_cols(
                             t[0], i.indices,
                             int(t[1].get_shape()
                                 [0 if t[1].get_shape().ndims == 1 else 1]))
                         for i, t in zip(self.inputs, tuples))

    @abstractmethod
    def _compute_mpe_path(self, counts, *input_values):
        """Assemble TF operations computing the MPE branch counts for each input
        of the node.

        To be re-implemented in sub-classes.

        Args:
            counts (Tensor): Branch counts for each output value of this node.
            *input_values (Tensor): For each input, a tensor containing the value
                                    or log value produced by the input node. Can
                                    be ``None`` if the input is not connected.

        Returns:
            list of Tensor: For each input, branch counts to pass to the node
            connected to the input. Each tensor is of shape ``[None, out_size]``,
            where the first dimension corresponds to the batch size and the
            second dimension is the size of the output of the input node.
        """

    @abstractmethod
    def _compute_log_mpe_path(self, counts, *input_values):
        """Assemble TF operations computing the MPE branch counts for each input
        of the node assuming that value is computed in log space.

        To be re-implemented in sub-classes.

        Args:
            counts (Tensor): Branch counts for each output value of this node.
            *input_values (Tensor): For each input, a tensor containing the value
                                    or log value produced by the input node. Can
                                    be ``None`` if the input is not connected.

        Returns:
            list of Tensor: For each input, branch counts to pass to the node
            connected to the input. Each tensor is of shape ``[None, out_size]``,
            where the first dimension corresponds to the batch size and the
            second dimension is the size of the output of the input node.
        """

    # @abstractmethod
    # def _compute_gradient(self, gradients, *input_values):
    #     """Assemble TF operations computing gradients for each input of the node.
    #
    #     To be re-implemented in sub-classes.
    #
    #     Args:
    #         counts (Tensor): Branch counts for each output value of this node.
    #         *input_values (Tensor): For each input, a tensor containing the value
    #                                 or log value produced by the input node. Can
    #                                 be ``None`` if the input is not connected.
    #
    #     Returns:
    #         list of Tensor: For each input, branch counts to pass to the node
    #         connected to the input. Each tensor is of shape ``[None, out_size]``,
    #         where the first dimension corresponds to the batch size and the
    #         second dimension is the size of the output of the input node.
    #     """


class VarNode(Node):
    """An abstract class defining a variable node of the SPN graph.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        name (str): Name of the node.
    """

    def __init__(self, feed=None, name=None):
        super().__init__(InferenceType.MARGINAL, name)
        self.attach_feed(feed)

    @utils.docinherit(Node)
    @abstractmethod
    def deserialize(self, data):
        super().deserialize(data)
        self.attach_feed(None)

    def attach_feed(self, feed):
        """Set a tensor that feeds this node.

        Args:
           feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                          an internal placeholder will be used to feed this
                          node.
        """
        if feed is None:
            self._feed = self._placeholder
        else:
            self._feed = feed

    @property
    def feed(self):
        """Tensor: Tensor feeding this node."""
        return self._feed

    @utils.docinherit(Node)
    def _create(self):
        self._placeholder = self._create_placeholder()

    @abstractmethod
    def _create_placeholder(self):
        """Create a placeholder that will be used to feed this variable node
        when no other feed is available.

        To be re-implemented in a sub-class.

        Returns:
            Tensor: A TF placeholder.
        """

    def _const_out_size(self):
        """bool: If True, the number of outputs of this node does not depend
        on the inputs of the node and is fixed.

        Variable nodes always have a fixed number of outputs.
        """
        return True

    @abstractmethod
    def _compute_out_size(self):
        """Compute the size of the output of this node.

        To be re-implemented in sub-classes.

        Returns:
            int: Size of the output of this node.
        """

    @abstractmethod
    def _compute_scope(self):
        """Compute the scope of each output value of this node.

        To be re-implemented in sub-classes.

        Returns:
            list of Scope: A list of length ``out_size`` containing scopes of
            all output values of this node.
        """

    def _compute_valid(self):
        """Check for validity of the SPN rooted in this node. If the node has
        multiple outputs, it is considered valid if all outputs of that node
        come from a valid SPN.

        If valid, return the scope of each output value of this node, otherwise,
        return ``None`` to indicate that the node/SPN is not valid.

        Since a variable node is assumed to always be valid, this just returns
        the scope of the outputs of this node.

        Returns:
            list of Scope: A list of length ``out_size`` containing scopes of
            all output of this node.
        """
        return self._compute_scope()

    @abstractmethod
    def _compute_value(self):
        """Assemble TF operations computing the marginal value of this node.

        To be re-implemented in sub-classes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """

    def _compute_log_value(self):
        """Assemble TF operations computing the marginal log value of this node.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """
        return tf.log(self._compute_value())

    def _compute_mpe_value(self):
        """Assemble TF operations computing the MPE value of this node.

        The MPE value is equal to marginal value for VarNodes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """
        return self._compute_value()

    def _compute_log_mpe_value(self):
        """Assemble TF operations computing the log MPE value of this node.

        The MPE log value is equal to marginal log value for VarNodes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """
        return self._compute_log_value()

    @abstractmethod
    def _compute_mpe_state(self, counts):
        """Assemble TF operations computing the MPE state of the variables
        represented by the node.

        To be re-implemented in sub-classes.

        Args:
            counts (Tensor): Branch counts for each output value of this node.

        Returns:
            Tensor: MPE state of every variable in the node.
        """

    def _as_graph_element(self):
        """Used by TF to convert this class to a tensor.

        A class implementing this method can be used as a key in TF feeds
        (feed_dict) when running the graph.

        Returns:
            Variable or Tensor: A TF placeholder or variable representing
            this node.
        """
        return self._feed


class ParamNode(Node):
    """An abstract class defining a node parameterizing another node in the SPN
    graph.

    Args:
        name (str): Name of the node.
    """

    def __init__(self, name=None):
        super().__init__(InferenceType.MARGINAL, name)

    @abstractmethod
    def deserialize(self, data):
        """Initialize this node with the ``data`` dict during deserialization.

        Return a TF operation that must be executed to complete deserialization.

        Args:
            data (dict): Dictionary with all the data to be deserialized.

        Returns:
            TF operation used to finalize deserialization.
        """
        super().deserialize(data)
        return None

    def _const_out_size(self):
        """bool: If True, the number of outputs of this node does not depend
        on the inputs of the node and is fixed.

        Parameter nodes always have a fixed number of outputs.
        """
        return True

    @abstractmethod
    def _compute_out_size(self):
        """Compute the size of the output of this node.

        To be re-implemented in sub-classes.

        Returns:
            int: Size of the output of this node.
        """

    def _compute_scope(self):
        """Compute the scope of each output value of this node.

        Returns ``None``, since param nodes do not include variables.
        """
        return None

    def _compute_valid(self):
        """Check for validity of the SPN rooted in this node. If the node has
        multiple outputs, it is considered valid if all outputs of that node
        come from a valid SPN.

        If valid, return the scope of each output value of this node, otherwise,
        return ``None`` to indicate that the node/SPN is not valid.

        Returns ``None``, since param nodes do not include variables and do not
        affect validity.
        """
        return None

    @abstractmethod
    def _compute_value(self):
        """Assemble TF operations computing the marginal value of this node.

        To be re-implemented in sub-classes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """

    @abstractmethod
    def _compute_log_value(self):
        """Assemble TF operations computing the marginal log value of this node.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """

    def _compute_mpe_value(self):
        """Assemble TF operations computing the MPE value of this node.

        The MPE value is equal to marginal value for ParamNodes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """
        return self._compute_value()

    def _compute_log_mpe_value(self):
        """Assemble TF operations computing the log MPE value of this node.

        The MPE log value is equal to marginal log value for ParamNodes.

        Returns:
            Tensor: A tensor of shape ``[None, out_size]``, where the first
            dimension corresponds to the batch size.
        """
        return self._compute_log_value()

    @abstractmethod
    def _compute_hard_em_update(self, counts):
        """Assemble TF operations computing the hard EM update of the parameters
        of the node.

        To be re-implemented in sub-classes.

        Args:
            counts (Tensor): Branch counts for each output value of this node.

        Returns:
            Update operation.
        """


class DistributionNode(VarNode, abc.ABC):

    def __init__(self, feed=None, num_vars=1, name="DistributionLeaf"):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        self._num_vars = num_vars
        super().__init__(feed, name)

    @property
    def evidence(self):
        return self._evidence_indicator

    def _create_placeholder(self):
        return tf.placeholder(conf.dtype, [None, self._num_vars])

    def _create_evidence_indicator(self):
        return tf.placeholder_with_default(
            tf.cast(tf.ones_like(self._placeholder), tf.bool), shape=[None, self._num_vars])

    def _create(self):
        super()._create()
        self._evidence_indicator = self._create_evidence_indicator()


class ParameterizedDistributionNode(DistributionNode, abc.ABC):

    Accumulate = namedtuple("Accumulate", ["name", "shape", "init"])

    def __init__(self, accumulates=None, feed=None, num_vars=1, trainable=True,
                 name="ParameterizedDistribution"):
        self._variables = OrderedDict()
        self._accumulates = accumulates
        self._trainable = trainable
        super().__init__(feed, num_vars, name)

    @property
    def initialize(self):
        return tf.group(*[var.initializer for var in self._variables])

    @property
    def variables(self):
        return self._variables

    @property
    def accumulates(self):
        return self._accumulates

    @abc.abstractmethod
    def _compute_hard_em_update(self, counts):
        """Compute hard EM update for all variables contained in this node. """

    def _create(self):
        super()._create()
        self._variables = OrderedDict()
        for name, shape, init in self._accumulates:
            init_val = utils.broadcast_value(init, shape, dtype=conf.dtype)
            self._variables[name] = tf.Variable(
                init_val, dtype=conf.dtype, collections=['spn_distribution_accumulates'])

    # @abc.abstractmethod
    # def assign(self, accum, ):
    #     """Assign new values to variables based on accum """
        # assignment_ops = []
        # if values and named_values:
        #     raise ValueError(
        #         "Cannot specify both keyword arguments for values and names for values.")
        # if values:
        #     if len(values) != len(self._variables):
        #         raise StructureError(
        #             "{}: number of assignment values does not match the number of parameters. Got "
        #             "{}, expected {}.".format(self.name, len(values), len(self._variables)))
        #     for var, val in zip(self._variables.values(), values):
        #         assignment_ops.append(tf.assign(var, val))
        # if named_values:
        #     if len(named_values) != len(self._variables):
        #         raise StructureError(
        #             "{}: number of assignment values does not match the number of parameters. Got "
        #             "{}, expected {}.".format(self.name, len(values), len(self._variables)))
        #     for name, val in named_values.items():
        #         assignment_ops.append(tf.assign(self._variables[name], val))

