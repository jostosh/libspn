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


def compute_graph_up_dynamic(root, interface_nodes, template_val_fun, top_val_fun, max_len,
                             interface_heads, const_fun=None, all_values=None):

    # TODO interface arrays are currently not used
    interface_arrays = [
        tf.TensorArray(dtype=conf.dtype, size=max_len, name=r.name + "InterfaceArray")
        for r in interface_nodes]

    # The top array
    top_array = tf.TensorArray(dtype=conf.dtype, size=max_len, name="TopArray")

    def compute_single_step(t, interface_values, interface_arrays, top_val, top_array):
        all_values_step = {}

        # Wrapper for the value function that takes in the time step and the interface values
        val_fun = template_val_fun(t, interface_values)

        # First we construct the interface node tensors
        interface_values = [compute_graph_up(
            interface_node, val_fun=val_fun, const_fun=const_fun, all_values=all_values_step)
            for interface_node in interface_nodes]

        # If we call compute_graph_up on the head, it will take the values that are already
        # determined in the compute_graph_up call on the interface node (they should be in the
        # all_values_step dict
        heads = [compute_graph_up(head, val_fun=val_fun, const_fun=const_fun,
                                  all_values=all_values_step) for head in interface_heads]

        # Function wrapper for top value
        val_fun = top_val_fun(interface_values)

        # Then, we call compute_graph_up on the actual root of the SPN
        top_val = compute_graph_up(root, val_fun, const_fun=const_fun, all_values=all_values_step)

        # Write the value to the top array
        top_array = top_array.write(t, top_val)

        # Increment and return values
        return t + 1, heads, interface_arrays, top_val, top_array

    # TODO try if the first step can also be done in the while loop
    # Compute first step
    _, interface_init, interface_arrays, top_val, top_array = compute_single_step(
        0, None, interface_arrays, None, top_array)

    # TODO if the first step is done change start step to 0
    # Compute remaining steps
    step = tf.constant(1)
    _, final_val, interface_arrays, top_val, top_array = tf.while_loop(
        cond=lambda i, *_: tf.less(i, max_len),
        body=compute_single_step,
        loop_vars=[step, interface_init, interface_arrays, top_val, top_array])

    # Stack the interfaces
    interface_per_step = [arr.stack() for arr in interface_arrays]

    # Stack the values of the top array
    top_per_step = top_array.stack()
    return top_val, top_per_step, interface_per_step


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
