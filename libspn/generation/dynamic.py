
from collections import deque
from libspn import utils
from libspn.graph.node import Input, VarNode
from libspn.graph.sum import Sum
from libspn.graph.ivs import IVs
from libspn.graph.weights import Weights
from libspn.graph.contvars import ContVars
from libspn.graph.product import Product
from libspn.log import get_logger
from libspn.exceptions import StructureError
from copy import deepcopy
from libspn.utils.enum import Enum
import tensorflow as tf
from collections import OrderedDict
import random
from libspn.graph.basesum import BaseSum
from libspn.graph.parsums import ParSums
from libspn.graph.node import DynamicInterface, Input
from libspn.graph.productslayer import ProductsLayer
from libspn.graph.localsum import LocalSum
from libspn.graph.convsum import ConvSum
from libspn.graph.spatialpermproducts import SpatialPermProducts
from libspn.graph.convprod2d import ConvProd2D
import itertools
from libspn.generation.spatial import ConvSPN


logger = get_logger()


def dense_dynamic_interface(root, num_intf_mixtures):
    if not isinstance(root, BaseSum):
        raise TypeError("Root should be Sum, SumsLayer or ParSums instance.")
    if not all(isinstance(v.node, ProductsLayer) for v in root.values) or \
            not all(root.values[0].node == v.node for v in root.values):
        raise StructureError("Root should only have identical ProductsLayer children. Now root has "
                             "{} children of type {}.".format(len(root.values),
                                                              [type(r.node) for r in root.values]))
    penultimate = root.values[0].node
    interface = DynamicInterface(name="Interface", source=penultimate)
    interface_mixtures = ParSums(
        interface, num_sums=num_intf_mixtures, name="DenseInterfaceMixtures", interface_head=True)

    # Reconnect the ProductsLayer densely with the interface head
    prod_sizes = penultimate.num_or_size_prods
    new_inputs, new_prod_sizes = [], []

    # The number of inputs per product
    # TODO this seems very brittle. Probably this becomes more robust once the Operations are
    # modeled explicitly.
    inputs_per_prod = len(penultimate.inputs) // len(prod_sizes)

    # For each interface mixtures, we connect with all products in the ProductsLayer, effectively
    # increasing the output size from `num_prods` to `num_prods * num_intf_mixtures`
    for j in range(num_intf_mixtures):
        offset = 0
        for i, size in enumerate(prod_sizes):
            new_inputs.extend(
                penultimate.inputs[i * inputs_per_prod:(i + 1) * inputs_per_prod])
            new_inputs.append(Input(node=interface_mixtures, indices=j))
            new_prod_sizes.append(size + 1)
            offset += size
    penultimate.set_values(*new_inputs)
    penultimate.set_prod_sizes(new_prod_sizes)

    # Set the new sizes for the interface mixtures
    interface_mixtures._reset_sum_sizes(
        sum_sizes=[num_intf_mixtures * len(prod_sizes)] * num_intf_mixtures)
    root.set_values(penultimate)

    # Reconfigure scope
    interface._set_scope(interface.get_scope() * num_intf_mixtures)
    return interface, interface_mixtures


def conv_dyn_lattice(
        dynamic_vars, spatial_dims, num_sums_wicker, num_sums_interface, num_prods_wicker,
        num_prods_interface, dense_gen):
    conv_spn = ConvSPN()
    # TODO USE NORMAL STACK INSTEAD, OR DEAL WITH PADDING!
    conv_spn.add_stack(
        dynamic_vars, sum_num_channels=num_sums_wicker, spatial_dims=spatial_dims,
        prod_num_channels=num_prods_wicker, stack_size=4)
    
    child_temporal_prod = None
    root_prods = []
    for template_node in conv_spn.prod_nodes:
        # Auxiliary mixtures connected to the product at this level
        template_mixture = LocalSum(
            template_node, num_channels=num_sums_interface,
            grid_dim_sizes=template_node.output_shape_spatial[:2],
            name="TemplateMixture")
        spatial_dims = template_mixture.output_shape_spatial[:2]
        
        # Temporal products, combining information from previous time steps and current time step
        # locally
        temporal_prod = SpatialPermProducts(
            template_mixture, template_mixture, grid_dim_sizes=spatial_dims, name="TemporalProd")
        
        # Interface node that takes the output of the temporal product at the previous time step
        interface = DynamicInterface(source=temporal_prod)
        
        # Mixtures on top of the interface node
        interface_mixture = LocalSum(interface, interface_head=True, name="InterfaceMixture",
                                     num_channels=num_sums_interface, grid_dim_sizes=spatial_dims)
        
        # Reconfigure the temporal product input nodes
        # TODO this is still something to improve!
        temporal_prod.set_values(template_mixture, interface_mixture)
    
        if child_temporal_prod is not None:
            spatial_dims = child_temporal_prod.output_shape_spatial[:2]
            # TODO take over the properties of current prod_node
            temporal_mixture = LocalSum(
                child_temporal_prod, num_channels=num_sums_interface, grid_dim_sizes=spatial_dims,
                name="TemporalMixture")
            parent_prod = ConvProd2D(
                temporal_mixture, num_channels=num_prods_interface, grid_dim_sizes=spatial_dims,
                name="ParentProduct")
            spatial_dims = parent_prod.output_shape_spatial[:2]
            cross_mixture = LocalSum(
                parent_prod, temporal_prod, num_channels=num_sums_interface,
                grid_dim_sizes=spatial_dims, name="CrossMixture")
            dense_root = dense_gen.generate(cross_mixture)
            root_prods.append(dense_root.values[0].node)
            
        child_temporal_prod = temporal_prod

    root = Sum(*root_prods, name="Root")

    return root


def dynamic_wicker(
        conv_spn, spatial_dims, num_sums_wicker, num_sums_interface, num_prods_wicker,
        num_prods_interface, dense_gen):
    aux_conv_spn = ConvSPN()

    child_temporal_prod = None
    root_prods = []
    prod_nodes = conv_spn.prod_nodes
    for i, template_node in enumerate(conv_spn.prod_nodes):
        # Auxiliary mixtures connected to the product at this level
        template_mixture = LocalSum(
            template_node, num_channels=num_sums_interface,
            grid_dim_sizes=template_node.output_shape_spatial[:2],
            name="TemplateMixture")
        spatial_dims = template_mixture.output_shape_spatial[:2]

        # Temporal products, combining information from previous time steps and current time step
        # locally
        temporal_prod = SpatialPermProducts(
            template_mixture, template_mixture, grid_dim_sizes=spatial_dims, name="TemporalProd")

        # Interface node that takes the output of the temporal product at the previous time step
        interface = DynamicInterface(source=temporal_prod)

        # Mixtures on top of the interface node
        interface_mixture = LocalSum(interface, interface_head=True, name="InterfaceMixture",
                                     num_channels=num_sums_interface, grid_dim_sizes=spatial_dims)

        # Reconfigure the temporal product input nodes
        # TODO this is still something to improve!
        temporal_prod.set_values(template_mixture, interface_mixture)

        if child_temporal_prod is not None:
            spatial_dims = child_temporal_prod.output_shape_spatial[:2]
            # TODO take over the properties of current prod_node
            temporal_mixture = LocalSum(
                child_temporal_prod, num_channels=num_sums_interface, grid_dim_sizes=spatial_dims,
                name="TemporalMixture")
            parent_prod = ConvProd2D(
                temporal_mixture, num_channels=num_prods_interface, grid_dim_sizes=spatial_dims,
                strides=template_node._strides, dilation_rate=template_node._dilation_rate,
                pad_left=template_node._pad_left, pad_right=template_node._pad_right,
                pad_top=template_node._pad_top, pad_bottom=template_node._pad_bottom,
                name="ParentProduct")
            spatial_dims = parent_prod.output_shape_spatial[:2]
            cross_mixture = LocalSum(
                parent_prod, temporal_prod, num_channels = num_sums_interface,
                grid_dim_sizes = spatial_dims, name = "CrossMixture")

            pad_left = [node._pad_left for node in prod_nodes[i + 1:]]
            pad_right = [node._pad_right for node in prod_nodes[i + 1:]]
            pad_top = [node._pad_top for node in prod_nodes[i + 1:]]
            pad_bottom = [node._pad_bottom for node in prod_nodes[i + 1:]]

            dilation_rate = [node._dilation_rate for node in prod_nodes[i + 1:]]
            strides = [node._strides for node in prod_nodes[i + 1:]]

            print("Adding stack of size", len(prod_nodes) - (i + 1))
            sub_root = aux_conv_spn.add_stack(
                cross_mixture, spatial_dims=cross_mixture.output_shape_spatial[:2],
                pad_left=pad_left, pad_right=pad_right, pad_top=pad_top, pad_bottom=pad_bottom,
                dilation_rate=dilation_rate, strides=strides, stack_size=len(prod_nodes) - (i + 1))
            root_prods.append(sub_root)

        child_temporal_prod = temporal_prod
    root = Sum(*root_prods, name="Root")

    return root


class DynamicSPNComponent:

    class Step:

        def __init__(self, root_nodes, var_nodes, time):
            """Represents a single time step """
            self.time = time
            self.root_nodes = root_nodes
            self.var_nodes = var_nodes

    class InterfaceNode:

        def __init__(self, name, root_index, indices=None):
            """An interface is purely symbolic and acts as a placeholder while building DSPNs """
            self.name = name
            self.indices = indices
            self.root_index = root_index

    class TemplateNode:

        def __init__(self, type, name, inputs=None, weights=None, interface=None, **kwargs):
            """A template node can be seen as a Node factory, ready to make an instance of the
            given type and configuration at any time step. """
            self._type = type
            self._name = name
            self._kwargs = kwargs
            self._inputs = inputs
            self._weights = weights
            self._interface = interface

        def __str__(self):
            return self._name

        @property
        def interface_ancestor(self):
            """Whether this is an ancestor of an interface node """
            if self.takes_interface:
                return True
            if self.has_inputs:
                return any(node.interface_ancestor for node in self.inputs)
            return False

        @property
        def name(self):
            return self._name

        @property
        def has_inputs(self):
            return self._inputs is not None

        @property
        def inputs(self):
            return self._inputs

        def build(self, time, input_instances=None, interface_instances=None, is_first=False):
            """Creates a Node instance """
            input_tuples = []
            if self.has_inputs:
                if not input_instances:
                    raise StructureError("No input instances were defined")
                input_tuples.extend(
                    [(instance, node[1]) if isinstance(node, tuple) else instance
                     for node, instance in zip(self.inputs, input_instances)])
            if self.takes_interface:
                if interface_instances:
                    input_tuples.extend(
                        [(instance, interface_node.indices) if interface_node.indices else
                         instance for instance, interface_node in
                         zip(interface_instances, self.interface_nodes)])
                elif not is_first:
                    raise StructureError("This is not the first time step, requires an interface "
                                         "instance")

            node = self._type(*input_tuples, name=self._name + "_t{}".format(time), **self._kwargs)
            if self._weights:
                node.set_weights(self._weights)
            return node

        @property
        def takes_interface(self):
            """Whether this node has an interface node as input """
            return self._interface is not None

        @property
        def interface_nodes(self):
            """Returns the list of interface nodes connected to this TemplateNode """
            if not self.takes_interface:
                return []
            return [self._interface] if isinstance(
                self._interface, TemplateNetwork.InterfaceNode) else self._interface

        @property
        def interface_indices(self):
            """Returns the list of interface node indices """
            if not self.takes_interface:
                raise ValueError("This node does not take any interface nodes")
            return [interface_node.root_index for interface_node in self.interface_nodes]

    def __init__(self):
        self._nodes = {}
        self._feed_order = []
        self._child_nodes = {}
        self._root = []

    def add_node(self, type, name, is_root=None, inputs=None, weights=None, interface=None,
                 **kwargs):
        """Adds a TemplateNode to the TemplateNetwork. The methods below provide a more convenient
        interface for adding specific nodes. """
        if name in self._nodes:
            # Each template node must have a unique name
            raise StructureError("There already is a node with this name in the template network.")

        if inputs:
            # Wrap the inputs in a list
            inputs = [inputs] if isinstance(inputs, TemplateNetwork.TemplateNode) else inputs
            for inp in inputs:
                # Take out the actual TemplateNode if it resides in a (node, indices) tuple
                inp_node = inp[0] if isinstance(inp, tuple) else inp
                if not isinstance(inp_node, TemplateNetwork.TemplateNode):
                    raise TypeError("Input should also be a TemplateNode")

                if isinstance(self, TemplateNetwork):
                    # Validate that the input node is already part of the TemplateNetwork
                    if inp_node.name not in self._nodes:
                        raise StructureError("Input should already be added to the TemplateNetwork")

        # Create a new TemplateNode
        node = self._nodes[name] = TemplateNetwork.TemplateNode(
            type, name, inputs=inputs, weights=weights, interface=interface, **kwargs)
        # If this is a VarNode, we have to register it to the list of nodes to feed
        if issubclass(type, VarNode):
            self._feed_order.append(node)
        # If the given node is a root node (or interface) it should be registered as such
        if is_root:
            self._root.append(node)
        # Enrich the child nodes dict with the given node and its inputs
        self._child_nodes[node] = inputs
        return node

    def add_sum(self, name, inputs=None, weights=None, is_root=None, interface=None, **kwargs):
        """Adds a Sum TemplateNode """
        return self.add_node(Sum, name, is_root=is_root, inputs=inputs, weights=weights,
                             interface=interface, **kwargs)

    def add_product(self, name, inputs=None, is_root=None, interface=None, **kwargs):
        """Adds a Product TemplateNode """
        return self.add_node(Product, name, is_root=is_root, inputs=inputs, interface=interface,
                             weights=None, **kwargs)

    def add_ivs(self, name, is_root=None, **kwargs):
        """Adds an IVs TemplateNode """
        return self.add_node(IVs, name, inputs=None, is_root=is_root, weights=None, **kwargs)

    def add_cont_vars(self, name, is_root=None, **kwargs):
        """Adds a ContVars node """
        return self.add_node(ContVars, name, inputs=None, is_root=is_root, weights=None, **kwargs)


class TopNetwork(DynamicSPNComponent):

    def build(self, template_steps):
        """Builds the top network on top of the template steps. """
        if not self._root:
            # TODO this is pretty intuitive, but maybe not strictly necessary
            raise StructureError("TopNetwork must have a root before step construction can be "
                                 "performed.")
        steps = []
        for step in template_steps:
            root_instances = step.root_nodes
            t = step.time
            template_instance_map = {}

            def get_node_instance(template_node):
                input_instances = []
                if template_node.has_inputs:
                    # If this node has inputs, get their instances
                    for inp in template_node.inputs:
                        inp_node = inp[0] if isinstance(inp, tuple) else inp
                        input_instances.append(get_node_instance(inp_node))

                # If this template node has not be instantiated before, we do it now
                if template_node not in template_instance_map:
                    # First we need to collect the interface instances if there are any
                    if template_node.takes_interface:
                        interface_instances = [
                            root_instances[ind] for ind in template_node.interface_indices]
                    else:
                        interface_instances = None
                    # Now we do the actual node instantiation
                    instance = template_instance_map[template_node] = template_node.build(
                        time=t, input_instances=input_instances,
                        interface_instances=interface_instances)
                else:  # i.e. node was already instantiated
                    instance = template_instance_map[template_node]
                return instance

            # Get the root instances
            root_instances = [get_node_instance(r) for r in self._root]

            # Append a Step to the list
            steps.append(DynamicSPNComponent.Step(root_instances, step.var_nodes, t))
        return steps


class TemplateNetwork(DynamicSPNComponent):

    def __init__(self):
        super(TemplateNetwork, self).__init__()
        self._current_step = 0
        self._steps = {}

    def build_n_steps(self, n, start=0):
        prev_root = None
        steps = []
        for t in range(start, start + n):
            step_t = self.build_step(t=t, prev_root_instance=prev_root, is_first=t == start)
            prev_root = step_t.root_nodes
            steps.append(step_t)
        return steps

    def build_step(self, prev_root_instance, t=None, is_first=False):
        """Returns root node after constructing a single step """
        if not t:
            # TODO the current step logic is not very useful...
            t = self._current_step
            self._current_step += 1
        else:
            self._current_step = t + 1

        if not self._root:
            raise StructureError("TemplateNetwork must have a root before step construction can be "
                                 "performed.")

        template_instance_map = {}

        def get_node_instance(node):
            input_instances = []
            if node.has_inputs:
                for inp in node.inputs:
                    inp_node = inp[0] if isinstance(inp, tuple) else inp
                    # If this is our first step and the current node is an interface ancestor, we
                    # cannot instantiate it, so we skip it
                    if is_first and inp_node.interface_ancestor:
                        continue
                    input_instances.append(get_node_instance(inp_node))
            if node not in template_instance_map:
                # Otherwise, we take the interface node (if any) and instantiate a node for this
                # time step
                if node.takes_interface:
                    interface_instances = [
                        prev_root_instance[ind] for ind in node.interface_indices]
                else:
                    interface_instances = None
                instance = template_instance_map[node] = node.build(
                    time=t, input_instances=input_instances,
                    interface_instances=interface_instances, is_first=is_first)
            else:
                instance = template_instance_map[node]
            return instance

        # Get Node instance of the roots
        root_instances = [get_node_instance(r) for r in self._root]
        var_nodes = [template_instance_map[n] for n in self._feed_order]
        step = self._steps[t] = TemplateNetwork.Step(root_instances, var_nodes, t)
        return step

    def set_feed_order(self, order):
        if set(order) != set(self._feed_order):
            raise StructureError("The given feed order is not complete.")
        self._feed_order = order

    def copy(self):
        return deepcopy(self)


if __name__ == "__main__":
    template_network = TemplateNetwork()

    ix = template_network.add_ivs("iv_x", num_vars=1, num_vals=2)
    iy = template_network.add_ivs("iv_y", num_vars=1, num_vals=2)
    iz = template_network.add_ivs("iv_z", num_vars=1, num_vals=2)

    mixture_x0_w = Weights(num_weights=2, name="mixture_x0_w")
    mixture_x0 = template_network.add_sum("mix_x0", inputs=ix, weights=mixture_x0_w)
    mixture_x1_w = Weights(num_weights=2, name="mixture_x1_w")
    mixture_x1 = template_network.add_sum("mix_x1", inputs=ix, weights=mixture_x1_w)

    mixture_y0_w = Weights(num_weights=2, name="mixture_y0_w")
    mixture_y0 = template_network.add_sum("mix_y0", inputs=iy, weights=mixture_y0_w)
    mixture_y1_w = Weights(num_weights=2, name="mixture_y1_w")
    mixture_y1 = template_network.add_sum("mix_y1", inputs=iy, weights=mixture_y1_w)

    mixture_z0_w = Weights(num_weights=2, name="mixture_z0_w")
    mixture_z0 = template_network.add_sum("mix_z0", inputs=iz, weights=mixture_z0_w)
    mixture_z1_w = Weights(num_weights=2, name="mixture_z1_w")
    mixture_z1 = template_network.add_sum("mix_z1", inputs=iz, weights=mixture_z1_w)

    interface_forward_declaration = [TemplateNetwork.InterfaceNode("Interface0", 0),
                                     TemplateNetwork.InterfaceNode("Interface1", 1)]
    mixture_in0 = template_network.add_sum(
        "mix_interface0", interface=interface_forward_declaration)
    mixture_in1 = template_network.add_sum(
        "mix_interface1", interface=interface_forward_declaration)

    prod0 = template_network.add_product(
        "prod0", inputs=[mixture_x0, mixture_y0, mixture_z0, mixture_in0], is_root=True)
    prod1 = template_network.add_product(
        "prod1", inputs=[mixture_x1, mixture_y1, mixture_z1, mixture_in1], is_root=True)

    top_network = TopNetwork()
    top_weights = Weights(num_weights=2, name="top_w")
    top_root = top_network.add_sum("top_root", interface=interface_forward_declaration,
                                   is_root=True, weights=top_weights)

    template_steps = template_network.build_n_steps(3)
    top_steps = top_network.build(template_steps)


