from libspn.generation.spatial import ConvSPN
import libspn as spn
from libspn.log import get_logger
import numpy as np


logger = get_logger()


def _preprocess_prod_num_channels(*inp_nodes, prod_num_channels, kernel_size):
    kernel_surface = kernel_size ** 2 if isinstance(kernel_size, int) else np.prod(kernel_size)

    if not all(isinstance(node, (spn.GaussianLeaf, spn.IVs)) for node in inp_nodes):
        logger.warn("Preprocessing skipped. Preprocessing only works for IVs and "
                    "GaussianLeaf nodes.")
        return prod_num_channels
    first_num_channels = 1
    for node in inp_nodes:
        if isinstance(node, spn.IVs):
            first_num_channels *= node.num_vals ** kernel_surface
        else:
            first_num_channels *= node.num_components ** kernel_surface
    logger.warn("Replacing first number of prod channels '{}' with '{}', since there are "
                "no more possible permutations.".format(
        prod_num_channels[0], first_num_channels))
    return (first_num_channels,) + prod_num_channels[1:]


def full_wicker(
        *inp_nodes, spatial_dims=(28, 28), strides=(1, 2, 2, 1, 1),
        sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32, 32, 64, 64), prod_num_channels=(16, 32, 32, 64, 64),
        num_channels_top=32):
    conv_spn_gen = ConvSPN()

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    root = conv_spn_gen.full_wicker(
        *inp_nodes, sum_num_channels=sum_num_channels,
        prod_num_channels=prod_num_channels, spatial_dims=spatial_dims,
        num_channels_top=num_channels_top, strides=strides, kernel_size=kernel_size,
        sum_node_type=sum_node_types)
    return root


def dilate_stride_double_stride(
        *inp_nodes, spatial_dims=(28, 28), sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32), prod_num_channels=(16, 32),
        dense_gen=None):
    conv_spn_gen = ConvSPN()

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    dilate_stride0 = conv_spn_gen.add_dilate_stride(
        *inp_nodes, sum_num_channels=sum_num_channels,
        prod_num_channels=prod_num_channels, spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDilateStride",
        sum_node_type=(sum_node_types, 'skip'))
    double_stride0 = conv_spn_gen.add_double_stride(
        *inp_nodes, sum_num_channels=sum_num_channels,
        prod_num_channels=prod_num_channels, spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDoubleStride",
        sum_node_type=(sum_node_types, 'skip'))
    spatial_dims = double_stride0.output_shape_spatial[:2]
    if sum_node_types == 'local':
        dsds_mixtures_top = spn.LocalSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures_top = spn.ConvSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    if dense_gen is not None:
        return dense_gen.generate(dsds_mixtures_top)
    return dsds_mixtures_top


def dilate_stride_double_stride_full_wicker(
        *inp_nodes, spatial_dims=(28, 28), sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32), prod_num_channels=(16, 32), num_channels_top=32):
    conv_spn_gen = ConvSPN()

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    dilate_stride0 = conv_spn_gen.add_dilate_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDilateStride",
        sum_node_type=(sum_node_types, 'skip'))
    double_stride0 = conv_spn_gen.add_double_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDoubleStride",
        sum_node_type=(sum_node_types, 'skip'))
    spatial_dims = double_stride0.output_shape_spatial[:2]
    if sum_node_types == 'local':
        dsds_mixtures_top = spn.LocalSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures_top = spn.ConvSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")

    spatial_dims = dsds_mixtures_top.output_shape_spatial[:2]
    root = conv_spn_gen.full_wicker(
        dsds_mixtures_top, sum_num_channels=sum_num_channels[2:],
        prod_num_channels=prod_num_channels[2:], spatial_dims=spatial_dims,
        strides=1, kernel_size=kernel_size, num_channels_top=num_channels_top,
        sum_node_type=sum_node_types)
    return root


def double_dilate_stride_double_stride(
        *inp_nodes, spatial_dims=(28, 28), sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32), prod_num_channels=(16, 32),
        dense_gen=None):
    conv_spn_gen = ConvSPN()

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    dilate_stride0 = conv_spn_gen.add_dilate_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDilateStride",
        sum_node_type=(sum_node_types, 'skip'))
    double_stride0 = conv_spn_gen.add_double_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDoubleStride",
        sum_node_type=(sum_node_types, 'skip'))
    spatial_dims = double_stride0.output_shape_spatial[:2]

    if sum_node_types == 'local':
        dsds_mixtures = spn.LocalSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures = spn.ConvSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")

    pad_bottom = (4 - (spatial_dims[0] % 4), None)
    pad_right = (4 - (spatial_dims[1] % 4), None)

    dilate_stride1 = conv_spn_gen.add_dilate_stride(
        dsds_mixtures, sum_num_channels=sum_num_channels[2:],
        prod_num_channels=prod_num_channels[2:], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3STopDilateStride", pad_right=pad_right,
        pad_bottom=pad_bottom,
        sum_node_type=(sum_node_types, 'skip'))
    double_stride1 = conv_spn_gen.add_double_stride(
        dsds_mixtures, double_stride0, sum_num_channels=sum_num_channels[2:],
        prod_num_channels=prod_num_channels[2:], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3STopDoubleStride", pad_right=pad_right,
        pad_bottom=pad_bottom,
        sum_node_type=(sum_node_types, 'skip'))
    spatial_dims = double_stride1.output_shape_spatial[:2]

    if sum_node_types == 'local':
        dsds_mixtures_top = spn.LocalSum(
            dilate_stride1, double_stride1, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures_top = spn.ConvSum(
            dilate_stride1, double_stride1, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    if dense_gen is not None:
        return dense_gen.generate(dsds_mixtures_top)
    return dsds_mixtures_top


def wicker_dense(
        *inp_nodes, spatial_dims=(28, 28), strides=(1, 2, 2),
        sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32, 32), prod_num_channels=(16, 32, 32),
        wicker_stack_size=3, dense_gen=None):
    conv_spn_gen = ConvSPN()
    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)
    root = conv_spn_gen.wicker_stack(
        *inp_nodes, stack_size=wicker_stack_size, strides=strides,
        sum_num_channels=sum_num_channels, prod_num_channels=prod_num_channels,
        name_prefix="WickerDense", dense_generator=dense_gen, spatial_dims=spatial_dims,
        sum_node_type=sum_node_types)
    return root

