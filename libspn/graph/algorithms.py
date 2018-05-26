# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSPN graph algorithms."""

from collections import deque, defaultdict
import tensorflow as tf
import libspn.conf as conf
from collections import OrderedDict

from libspn.exceptions import StructureError


def compute_graph_up_dynamic(root, val_fun_step, interface_init, const_fun=None,
                             all_values=None):

    # Batch size is needed to generate the 'no evidence' values for interface nodes
    batch_size = root.get_batch_size()
    max_len = root.get_maxlen()

    # Interface nodes should be initialized at t == 0
    interface_nodes = []
    traverse_graph(root, lambda node: interface_nodes.append(node) if node.is_interface else None)

    # Interface inits
    interface_val_t0 = []
    with tf.name_scope("InterfaceInit"):
        for node in interface_nodes:
            out_size = node.source.get_out_size()
            shape = (batch_size, out_size) if isinstance(out_size, int) else (batch_size,) + out_size
            interface_val_t0.append(interface_init(
                shape=shape, dtype=conf.dtype, name=node.name + "Init"))

    # Get all nodes
    nodes = []
    traverse_graph(root, lambda node, *_: nodes.append(node))

    with tf.name_scope("ValueArrayInit"):
        value_arrays = [tf.TensorArray(size=max_len, dtype=conf.dtype, clear_after_read=False,
                                       name=node.name + "ValueArray")
                        for node in nodes]

    def compute_single_step(t, interface_values, value_arrays, top_val):

        # Wrapper for the value function that takes in the time step and the interface values
        all_values_step = {}
        interface_value_map = {node: val for node, val in zip(interface_nodes, interface_values)}
        val_fun = val_fun_step(t, interface_value_map)

        # Then, we call compute_graph_up on the actual root of the SPN
        top_val = compute_graph_up(root, val_fun, const_fun=const_fun, all_values=all_values_step)

        # Get the values of the interface nodes from their sources in the current step. These
        # should be used during the next iteration
        for i, node in enumerate(interface_nodes):
            interface_values[i] = all_values_step[node.source]

        with tf.name_scope("WriteValuesToArrays"):
            for node, value_tensor in all_values_step.items():
                node_ind = nodes.index(node)
                value_arrays[node_ind] = value_arrays[node_ind].write(t, value_tensor)

        # Increment and return values
        return t + 1, interface_values, value_arrays, top_val

    with tf.name_scope("RootInit"):
        top_size = root.get_out_size()
        shape = (batch_size, top_size) if isinstance(top_size, int) else (batch_size,) + top_size
        # Dummy tensor
        top_val_init = tf.zeros(shape=shape, dtype=conf.dtype)

    step = tf.constant(0)
    _, final_val, value_arrays, top_val = tf.while_loop(
        cond=lambda i, *_: tf.less(i, max_len),
        body=compute_single_step,
        loop_vars=[step, interface_val_t0, value_arrays, top_val_init])

    # Finally we assign the arrays to the dict
    for node, values in zip(nodes, value_arrays):
        all_values[node] = values

    # Stack the values of the top array
    # top_per_step = top_array.stack()
    top_per_step = all_values[root].stack()
    return top_val, top_per_step


def compute_graph_up(root, val_fun, const_fun=None, all_values=None):
    """Computes a certain value for the ``root`` node in the graph, assuming
    that for op nodes, the value depends on values produced by inputs of the op
    node. For this, it traverses the graph depth-first from the ``root`` node
    to the leaf nodes.

    Args:
        root (Node): The root of the SPN graph.
        val_fun (function): A function ``val_fun(node, *args)`` producing a
            certain value for the ``node``. For an op node, it will have
            additional arguments with values produced for the input nodes of
            ``node``.  The arguments will NOT be added if ``const_fun``
            returns ``True`` for the node. The arguments can be ``None`` if
            the input was empty.
        const_fun (function): A function ``const_fun(node)`` that should return
            ``True`` if the value generated by ``val_fun`` does not depend on
            the values generated for the input nodes, i.e. it is a constant
            function. If set to ``None``, it is assumed to always return
            ``False``, i.e. no ``val_fun`` is a constant function.
        all_values (dict): A dictionary indexed by ``node`` in which values
            computed for each node will be stored. Can be set to ``None``.

    Returns:
        The value for the ``root`` node.
    """
    if all_values is None:  # Dictionary of computed values indexed by node
        all_values = {}
    stack = deque()  # Stack of nodes to process
    stack.append(root)

    last_val = None
    while stack:
        next_node = stack[-1]
        # Was this node already processed?
        # This might happen if the node is referenced by several parents
        if next_node not in all_values:
            if next_node.is_op:
                # OpNode
                input_vals = []
                all_input_vals = True
                if const_fun is None or const_fun(next_node) is False:
                    # Gather input values for non-const val fun
                    for inpt in next_node.inputs:
                        if inpt:  # Input is not empty
                            try:
                                # Check if input_node in all_vals
                                input_vals.append(all_values[inpt.node])
                            except KeyError:
                                all_input_vals = False
                                stack.append(inpt.node)
                        else:
                            # This input was empty, use None as value
                            input_vals.append(None)
                # Got all inputs?
                if all_input_vals:
                    last_val = val_fun(next_node, *input_vals)
                    all_values[next_node] = last_val
                    stack.pop()
            else:
                # VarNode, ParamNode
                last_val = val_fun(next_node)
                all_values[next_node] = last_val
                stack.pop()
        else:
            stack.pop()

    return last_val


def compute_graph_up_down(root, down_fun, graph_input, up_fun=None,
                          up_values=None, down_values=None):
    """Computes a values for every node in the graph moving first up and then down
    the graph. When moving up, it behaves exactly as :meth:`compute_graph_up`.
    When moving down it computes values for each input of a node based on
    values produced for inputs of parent nodes connected to this node. For this,
    it traverses the graph breadth-first from the ``root`` node to the leaf nodes.

    Args:
        root (Node): The root of the SPN graph.
        down_fun (function): A function ``down_fun(node, parent_vals)``
            producing values for each input of the ``node``. The argument
            ``parent_vals`` is a list containing the values obtained for each
            parent node input connected to this node.
        graph_input: The value passed as a single parent value to the function
            computing the values for the root node or a function which computes
            that value.
        up_fun (function): A function ``up_fun(node, *args)`` producing a
            certain value for the ``node``. For an op node, it will have
            additional arguments with values produced for the input nodes of
            ``node``. The arguments can be ``None`` if the input was empty.
        up_values (dict): A dictionary indexed by ``node`` in which values
            computed for each node during the upward pass will be stored. Can
            be set to ``None``.
        down_values (dict): A dictionary indexed by ``node`` in which values
            computed for each input of a node during the downward pass will be
            stored. Can be set to ``None``.
    """
    if down_values is None:  # Dictionary of computed values indexed by node
        down_values = {}
    queue = deque()  # Queue of nodes with computed values, but unprocessed inputs
    parents = defaultdict(list)

    def up_fun_parents(node, *args):
        """Run up_fun and for each node find parent node inputs having the node
        connected."""
        # For each input, add the node and input number as relevant parent node
        # input to the connected node
        if node.is_op:
            for nr, inpt in enumerate(node.inputs):
                if inpt:
                    parents[inpt.node].append((node, nr))
        # Run up_fun
        if up_fun is not None:
            return up_fun(node, *args)

    # Traverse up
    compute_graph_up(root, val_fun=up_fun_parents, all_values=up_values)

    # Add root node
    if callable(graph_input):
        graph_input = graph_input()
    down_values[root] = down_fun(root, [graph_input])
    if root.is_op:
        queue.append(root)

    # Traverse down
    while queue:
        next_node = queue.popleft()
        children = set(i.node for i in next_node.inputs if i)
        for child in children:
            if child not in down_values:  # Not computed yet
                # Get all parent_vals
                parent_vals = []
                try:
                    for parent_node, parent_input_nr in parents[child]:
                        parent_vals.append(
                            down_values[parent_node][parent_input_nr])
                    # All parent values are available, compute value
                    down_values[child] = down_fun(child, parent_vals)
                    # Enqueue for further processing of children
                    if child.is_op:
                        queue.append(child)
                except KeyError:
                    # Not all parent values were available
                    pass


def compute_graph_up_down_dynamic(root, down_fun_step, graph_input_end, graph_input_default,
                                  reduce_parents_fun_step, reduce_init=None,
                                  reduce_binary_op=None):
    """Computes a values for every node in the graph moving first up and then down
    the graph. When moving up, it behaves exactly as :meth:`compute_graph_up`.
    When moving down it computes values for each input of a node based on
    values produced for inputs of parent nodes connected to this node. For this,
    it traverses the graph breadth-first from the ``root`` node to the leaf nodes.

    Args:
        root (Node): The root of the SPN graph.
        down_fun (function): A function ``down_fun(node, parent_vals)``
            producing values for each input of the ``node``. The argument
            ``parent_vals`` is a list containing the values obtained for each
            parent node input connected to this node.
        graph_input_end: The value passed as a single parent value to the function
            computing the values for the root node or a function which computes
            that value.
        up_fun (function): A function ``up_fun(node, *args)`` producing a
            certain value for the ``node``. For an op node, it will have
            additional arguments with values produced for the input nodes of
            ``node``. The arguments can be ``None`` if the input was empty.
        up_values (dict): A dictionary indexed by ``node`` in which values
            computed for each node during the upward pass will be stored. Can
            be set to ``None``.
        down_values (dict): A dictionary indexed by ``node`` in which values
            computed for each input of a node during the downward pass will be
            stored. Can be set to ``None``.
    """
    parents = defaultdict(list)

    def up_fun_parents(node, *args):
        """Run up_fun and for each node find parent node inputs having the node
        connected."""
        # For each input, add the node and input number as relevant parent node
        # input to the connected node
        if node.is_op:
            for nr, inpt in enumerate(node.inputs):
                if inpt:
                    parents[inpt.node].append((node, nr))

    # Traverse up for parents
    compute_graph_up(root, val_fun=up_fun_parents)

    # Add root node
    if callable(graph_input_end):
        graph_input_end = graph_input_end()
    if callable(graph_input_default):
        graph_input_default = graph_input_default()

    maxlen = root.get_maxlen()

    # We need to know the breadth-first order of the nodes
    node_order = []
    traverse_graph(root, fun=lambda node: node_order.append(node))

    if reduce_binary_op is None or reduce_init is None:
        if reduce_binary_op is not None or reduce_init is not None:
            raise ValueError("Must specify both binary reduce op and initializer or neither. "
                             "Now only one of the two was given.")
        # Initialize the arrays holding the values
        with tf.name_scope("DynamicValueArrayInit"):
            arrays_or_reduced_output = [tf.TensorArray(
                size=maxlen, clear_after_read=True, name=node.name + "DownArray",
                dtype=conf.dtype) for node in node_order]

        def reduce_step_or_write(t, val, new_val):
            return val.write(t, new_val)
    else:
        with tf.name_scope("DownValuesInit"):
            batch_size = root.get_batch_size()
            arrays_or_reduced_output = []
            for node in node_order:
                if node.is_param:
                    out_size = tuple(node.variable.shape.as_list())
                else:
                    out_size = node.get_out_size()
                    out_size = (out_size,) if isinstance(out_size, int) else out_size
                shape = (batch_size,) + out_size
                arrays_or_reduced_output.append(reduce_init(
                    shape=shape, name=node.name + "DownInit"))

        def reduce_step_or_write(_, val, new_val):
            return reduce_binary_op(val, new_val)

    # We need to know the sources in breadth-first order as well
    sources = []
    traverse_graph(root, lambda node: sources.append(node) if node.has_receiver else None)

    # For each of the sources, we will have an array storing the down values of its receiver
    # in the previous time step. For this to work, we need to pass an initial nested list with
    # dummy tensors, filled with zero in this case
    interface_sources_prev = []
    batch_size = root.get_batch_size()
    with tf.name_scope("SourceValuesPrevInit"):
        for source in sources:
            # Determine shape of dummy tensor
            size = source.get_out_size()
            shape = (batch_size, size) if isinstance(size, int) else (batch_size,) + size
            # TODO, maybe we don't need to have a separate tensor for each input

            # For each parent of the receiver, we should have the dummy tensor ready
            prev_down_for_this_source = []
            for _ in parents[source.receiver]:
                prev_down_for_this_source.append(tf.zeros(shape, dtype=conf.dtype))

            # Add the list of dummy tensors to the nested list
            interface_sources_prev.append(prev_down_for_this_source)

    def single_step(t, interface_sources_prev, arrays_or_reduced_output):
        """Defines a single time step in the MPE path calculation"""
        down_values = dict()

        # First, we feed the root of the graph with the graph input (which is optionally unique
        # for the last step)
        with tf.name_scope("SelectGraphInput"):
            root_parent_vals = [tf.cond(
                tf.equal(t, maxlen - 1), lambda: graph_input_end, lambda: graph_input_default)]

        # The values are then combined, mapping multiple parent tensors to a single one
        with tf.name_scope("CombineParents") as combine_parents_scope:
            root_reduced_val = reduce_parents_fun_step(t, root, root_parent_vals)

        # Write the first tensor to the value arrays
        with tf.name_scope("WriteValAtStep") as write_val_step_scope:
            arrays_or_reduced_output[0] = reduce_step_or_write(
                t, arrays_or_reduced_output[0], root_reduced_val)

        # Compute the tensors to pass down the graph
        with tf.name_scope("DownFun") as down_fun_scope:
            down_values[root] = down_fun_step(t, root, root_reduced_val)

        queue = deque()  # Queue of nodes with computed values, but unprocessed inputs
        # Initialize the queue
        if root.is_op:
            queue.append(root)

        # Traverse down
        while queue:
            next_node = queue.popleft()

            # Get unique children unique children
            children = set(i.node for i in next_node.inputs if i)

            for child in children:
                if child not in down_values:  # Not computed yet
                    # Collect the parent values
                    parent_vals = []
                    try:
                        # Go through list of parents of this child
                        for parent_node, parent_input_nr in parents[child]:
                            parent_vals.append(down_values[parent_node][parent_input_nr])

                        with tf.name_scope(combine_parents_scope):
                            if child.has_receiver:
                                # If this node has a receiver (it is a source), then we should take
                                # the parent values of its receiver in previous time step
                                # if t < max_steps - 1, otherwise just take the values of its own
                                # parent (which will be values coming from the top network)
                                parent_vals_prev = interface_sources_prev[sources.index(child)]

                                # This is where you can see that the prev_down_values tensors really
                                # don't have a meaning other than just pre-occupying memory for the
                                # while loop_vars
                                reduced_parents_val = tf.cond(
                                    tf.less(t, maxlen - 1),
                                    lambda: reduce_parents_fun_step(t, child, parent_vals_prev),
                                    lambda: reduce_parents_fun_step(t, child, parent_vals))
                            else:
                                # Combine value of parents to get the value of the node
                                # print()
                                reduced_parents_val = reduce_parents_fun_step(t, child, parent_vals)
                        with tf.name_scope(write_val_step_scope):
                            # Write the combined value to the array or accumulate directly
                            node_ind = node_order.index(child)
                            arrays_or_reduced_output[node_ind] = reduce_step_or_write(
                                t, arrays_or_reduced_output[node_ind], reduced_parents_val)
                        with tf.name_scope(down_fun_scope):
                            # Assemble tensors for the children given the reduced value
                            down_values[child] = down_fun_step(t, child, reduced_parents_val)

                        # Enqueue for further processing of children
                        if child.is_op:
                            queue.append(child)

                    except KeyError:
                        # Not all parent values were available
                        pass

        # Now we set the prev_down_values for the next step. For each source, we look at the
        # parents of the receiver. These have their down_values stored in the dict and will be
        # used in the next iteration.
        for s_ind, source in enumerate(sources):
            for inp_ind, (parent_node, parent_input_nr) in enumerate(parents[source.receiver]):
                interface_sources_prev[s_ind][inp_ind] = down_values[parent_node][parent_input_nr]

        # [[print(ne.shape) for ne in n] for n in interface_sources_prev]
        # [print(arr.name, arr.shape) for arr in arrays_or_reduced_output]
        return t - 1, interface_sources_prev, arrays_or_reduced_output

    # Execute the loop, from t == max_steps - 1 through t == 0
    # [[print(ne.shape) for ne in n] for n in interface_sources_prev]
    # [print(arr.name, arr.shape) for arr in arrays_or_reduced_output]
    step = tf.constant(maxlen - 1)
    _, _, arrays_or_reduced_output = tf.while_loop(
        cond=lambda t, *_: tf.greater_equal(t, 0),
        body=single_step,
        loop_vars=[step, interface_sources_prev, arrays_or_reduced_output],
        name="BackwardLoop"
    )


    return {node: arr for node, arr in zip(node_order, arrays_or_reduced_output)}


def traverse_graph(root, fun, skip_params=False):
    """Runs ``fun`` on descendants of ``root`` (including ``root``) by
    traversing the graph breadth-first until ``fun`` returns True.

    Args:
        root (Node): The root of the SPN graph.
        fun (function): A function ``fun(node)`` executed once for every node of
                        the graph. It should return ``True`` if traversing
                        should be stopped.
        skip_params (bool): If ``True``, the param nodes will not be traversed.

    Returns:
        Node: Returns the last traversed node (the one for which ``fun``
        returned True) or ``None`` if ``fun`` never returned ``True``.
    """
    visited_nodes = set()  # Set of visited nodes
    queue = deque()
    queue.append(root)

    while queue:
        next_node = queue.popleft()
        if next_node not in visited_nodes:
            if fun(next_node):
                return next_node
            visited_nodes.add(next_node)
            # OpNode?: enqueue inputs
            if next_node.is_op:
                for i in next_node.inputs:
                    if (i and  # Input not empty
                            not (skip_params and i.is_param)):
                        queue.append(i.node)

    return None


# def compute_graph_up_down_comb(root, down_fun, comb_fun, graph_input,
#                                up_fun=None, up_values=None, comb_values=None,
#                                down_values=None, up_skip_params=False,
#                                down_skip_params=False):
#     """This function behaves similarly to :meth:`compute_graph_up_down`, but it
#     combines values produced by the parents using ``combine_fun`` before
#     running ``down_fun`` if a node has more than one parent. If there is only
#     one parent, it passes the value produced by that parent directly to
#     ``down_fun``. The combined values are stored in ``down_values``.

#     It assumes that ``down_fun`` produces a list of values, one for each input
#     of the node, even if the inputs are from the same child node.

#     Args:
#         root (Node): The root of the SPN graph.
#         down_fun (function): A function ``down_fun(node, val)`` producing a
#             certain value for the ``node``. The argument ``val`` is the combined
#             value passed from the parent nodes of this node.
#         comb_fun (function): A function ``combine_fun(parent_vals)`` combining
#             values from multiple parents into a single output value. The argument
#             ``parent_vals`` is a dict indexed by a parent node containing the
#             values produced for the parent nodes.
#         graph_input: The value passed as a parent value to the function
#             computing the value for the root node. It is added to the
#             ``parent_vals`` dict as ``{None: graph_input}``.
#         up_fun (function): A function ``up_fun(node, *args)`` producing a
#             certain value for the ``node``. For an op node, it will have
#             additional arguments ``input_values`` and, optionally, when
#             ``up_skip_params`` is ``False``, ``param_input_values``.
#             ``input_values`` is a list of values generated for the input nodes
#             of this op node, while ``param_input_values`` is a list of values
#             generated for the parameter nodes of this op node.
#         up_values (dict): A dictionary indexed by ``node`` in which values
#             computed for each node during the upward pass will be stored. Can
#             be set to ``None``.
#         comb_values (dict): A dictionary indexed by ``node`` in which combined
#             parent values computed for each node during the downward pass will
#             be stored. Can be set to ``None``.
#         down_values (dict): A dictionary indexed by ``node`` in which values
#             computed for each node during the downward pass will be stored. Can
#             be set to ``None``.
#         up_skip_params (bool): If ``True``, the param nodes will not be
#             traversed during the upward pass.
#         down_skip_params (bool): If ``True``, the param nodes will not be
#             traversed during the downward pass.
#     """
#     def fun(node, all_parent_vals):
#         # Extract parent values to be passed to node
#         relevant_vals = []
#         for parent_node, parent_vals in all_parent_vals.items():
#             if parent_node is None:
#                 # This is for the root node, for which parent_vals is graph_input
#                 relevant_vals.append(parent_vals)
#             else:
#                 for val, (parent_child, _) in zip(parent_vals, parent_node.inputs):
#                     if parent_child is node:
#                         relevant_vals.append(val)
#         # Combine, only if more than 1 parent
#         if len(relevant_vals) > 2:
#             comb_val = comb_fun(relevant_vals)
#         else:
#             comb_val = relevant_vals[0]
#         if comb_values is not None:
#             comb_values[node] = comb_val
#         return down_fun(node, comb_val)

#     compute_graph_up_down(root, down_fun=fun, graph_input=graph_input,
#                           up_fun=up_fun, up_values=up_values,
#                           down_values=down_values, up_skip_params=up_skip_params,
#                           down_skip_params=down_skip_params)
