from libspn.graph.op.localsum import LocalSum
from libspn.graph.op.convprod2d import ConvProd2D
from libspn.graph.op.convproddepthwise import ConvProdDepthWise
from libspn.graph.op.parsums import ParSums
from libspn.graph.op.sum import Sum
import numpy as np
import libspn as spn


def rat_spn(in_var):
    x = spn.BlockRandomDecompositions(in_var, num_decomps=10)

    num_alterations = int(np.ceil(np.log2(in_var.num_vars)))
    for i in range(num_alterations - 1):
        layer_suffix = "_{}".format(i)
        x = spn.BlockPermuteProduct(x, num_subsets=2, name="Products" + layer_suffix)
        x = spn.BlockSum(x, num_sums_per_block=4, name="Sums" + layer_suffix)

    layer_suffix = "_{}".format(num_alterations - 1)
    x = spn.BlockPermuteProduct(x, num_subsets=2, name="Products" + layer_suffix)
    x = spn.BlockSum(x, num_sums_per_block=1, name="ClassRoots")
    x = class_roots = spn.BlockMergeDecomps(x, factor=10)
    root = spn.BlockRootSum(x, name="Root")
    return root, class_roots


def wicker_convspn_two_non_overlapping(
        in_var, num_channels_prod, num_channels_sums, num_classes=10, edge_size=28,
        first_depthwise=False, supervised=True):
    stack_size = int(np.ceil(np.log2(edge_size))) - 2

    if first_depthwise:
        prod0 = ConvProdDepthWise(
            in_var, padding='valid', kernel_size=2, strides=2,
            grid_dim_sizes=[edge_size, edge_size])
    else:
        prod0 = ConvProd2D(
            in_var, num_channels=num_channels_prod[0], padding='valid', kernel_size=2, strides=2,
            grid_dim_sizes=[edge_size, edge_size])
    sum0 = LocalSum(prod0, num_channels=num_channels_sums[0])
    prod1 = ConvProd2D(sum0, padding='valid', kernel_size=2, strides=2, num_channels=num_channels_prod[1])
    h = LocalSum(prod1, num_channels=num_channels_sums[1])

    for i in range(stack_size):
        dilation_rate = 2 ** i
        h = ConvProd2D(
            h, padding='full', kernel_size=2, strides=1, dilation_rate=dilation_rate,
            num_channels=num_channels_prod[1])
        h = LocalSum(h, num_channels=num_channels_sums[2 + i])

    full_scope_prod = ConvProdDepthWise(
        h, padding='final', kernel_size=2, strides=1, dilation_rate=2 ** stack_size)
    if supervised:
        class_roots = ParSums(full_scope_prod, num_sums=num_classes)
        root = Sum(class_roots)
        return root, class_roots

    return Sum(full_scope_prod), None


def full_wicker(in_var, num_channels_prod, num_channels_sums, num_classes=10, edge_size=28,
                first_depthwise=False, supervised=True):
    stack_size = int(np.ceil(np.log2(edge_size))) - 1

    if first_depthwise:
        prod0 = ConvProdDepthWise(
            in_var, padding='full', kernel_size=2, strides=1,
            grid_dim_sizes=[edge_size, edge_size])
    else:
        prod0 = ConvProd2D(
            in_var, num_channels=num_channels_prod[0], padding='full', kernel_size=2, strides=1,
            grid_dim_sizes=[edge_size, edge_size])
    h = LocalSum(prod0, num_channels=num_channels_sums[0])

    for i in range(stack_size):
        dilation_rate = 2 ** (i + 1)
        h = ConvProdDepthWise(
            h, padding='full', kernel_size=2, strides=1, dilation_rate=dilation_rate)
        h = LocalSum(h, num_channels=num_channels_sums[1 + i])

    full_scope_prod = ConvProdDepthWise(
        h, padding='final', kernel_size=2, strides=1, dilation_rate=2 ** (stack_size + 1))
    if supervised:
        class_roots = ParSums(full_scope_prod, num_sums=num_classes)
        root = Sum(class_roots)
        return root, class_roots
    return Sum(full_scope_prod), None