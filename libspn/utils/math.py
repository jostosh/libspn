# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSPN math functions."""

import tensorflow as tf
import numpy as np
from libspn import conf
from libspn.ops import ops
from libspn.utils.serialization import register_serializable


class ValueType:

    """A class specifying various types of values that be passed to the SPN
    graph."""

    @register_serializable
    class RANDOM_UNIFORM:

        """A random value from a uniform distribution.

        Attributes:
            min_val: The lower bound of the range of random values.
            max_val: The upper bound of the range of random values.
        """

        def __init__(self, min_val=0, max_val=1):
            self.min_val = min_val
            self.max_val = max_val

        def __repr__(self):
            return ("ValueType.RANDOM_UNIFORM(min_val=%s, max_val=%s)" %
                    (self.min_val, self.max_val))

        def serialize(self):
            return {'min_val': self.min_val,
                    'max_val': self.max_val}

        def deserialize(self, data):
            self.min_val = data['min_val']
            self.max_val = data['max_val']


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor or values of a 1D tensor.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A 1D integer array.
        name (str): A name for the operation (optional).

    Returns:
        Tensor: Has the same dtype and number of dimensions and type as ``params``.
    """
    with tf.name_scope(name, "gather_cols", [params, indices]):
        params = tf.convert_to_tensor(params, name="params")
        indices = np.asarray(indices)
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D")
        # We need the size defined for optimizations
        if param_size is None:
            raise RuntimeError("The indexed dimension of 'params' is not specified")
        # Check indices
        if indices.ndim != 1:
            raise ValueError("'indices' must be 1D")
        if indices.size < 1:
            raise ValueError("'indices' cannot be empty")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("'indices' must be integer, not %s"
                             % indices.dtype)
        if np.any((indices < 0) | (indices >= param_size)):
            raise ValueError("'indices' must fit the the indexed dimension")
        # Define op
        if param_size == 1:
            # Single column tensor, indices must include it, just forward tensor
            return params
        elif indices.size == param_size and np.all(np.ediff1d(indices) == 1):
            # Indices index all params in the original order, pass through
            return params
        elif indices.size == 1:
            # Gathering a single column
            if param_dims == 1:
                # Gather is faster than custom for 1D.
                # It is as fast as slice for int64, and generates smaller graph
                return tf.gather(params, indices)
            else:
                if conf.custom_gather_cols:
                    return ops.gather_cols(params, indices)
                else:
                    return tf.slice(params, [0, indices[0]], [-1, 1])
        else:
            # Gathering multiple columns from multi-column tensor
            if param_dims == 1:
                # Gather is faster than custom for 1D.
                return tf.gather(params, indices)
            else:
                if conf.custom_gather_cols:
                    return ops.gather_cols(params, indices)
                else:
                    return tf.transpose(tf.gather_nd(tf.transpose(params),
                                                     np.expand_dims(indices, 1)))


def scatter_cols(params, indices, num_out_cols, name=None):
    """Scatter columns of a 2D tensor or values of a 1D tensor into a tensor
    with the same number of dimensions and ``num_out_cols`` columns or values.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A 1D integer array indexing the columns in the
                              output array to which ``params`` is scattered.
        num_cols (int): The number of columns in the output tensor.
        name (str): A name for the operation (optional).

    Returns:
        Tensor: Has the same dtype and number of dimensions as ``params``.
    """
    with tf.name_scope(name, "scatter_cols", [params, indices]):
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = np.asarray(indices)
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D")
        # We need the size defined for optimizations
        if param_size is None:
            raise RuntimeError("The indexed dimension of 'params' is not specified")
        # Check num_out_cols
        if not isinstance(num_out_cols, int):
            raise ValueError("'num_out_cols' must be integer, not %s"
                             % type(num_out_cols))
        if num_out_cols < param_size:
            raise ValueError("'num_out_cols' must be larger than the size of "
                             "the indexed dimension of 'params'")
        # Check indices
        if indices.ndim != 1:
            raise ValueError("'indices' must be 1D")
        if indices.size != param_size:
            raise ValueError("Sizes of 'indices' and the indexed dimension of "
                             "'params' must be the same")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("'indices' must be integer, not %s"
                             % indices.dtype)
        if np.any((indices < 0) | (indices >= num_out_cols)):
            raise ValueError("'indices' must be smaller than 'num_out_cols'")
        if len(set(indices)) != len(indices):
            raise ValueError("'indices' cannot contain duplicates")
        # Define op
        if num_out_cols == 1:
            # Scatter to a single column tensor, it must be from 1 column
            # tensor and the indices must include it. Just forward the tensor.
            return params
        elif num_out_cols == indices.size and np.all(np.ediff1d(indices) == 1):
            # Output equals input
            return params
        elif param_size == 1:
            # Scatter a single column tensor to a multi-column tensor
            if param_dims == 1:
                # Just pad with zeros, pad is fastest and offers smallest graph
                return tf.pad(params, [[indices[0], num_out_cols - indices[0] - 1]])
            else:
                # Currently pad is fastest (for GPU) and builds smaller graph
                # if conf.custom_scatter_cols:
                #     return ops.scatter_cols(
                #         params, indices,
                #         pad_elem=tf.constant(0, dtype=params.dtype),
                #         num_out_col=num_out_cols)
                # else:
                return tf.pad(params, [[0, 0],
                                       [indices[0], num_out_cols - indices[0] - 1]])
        else:
            # Scatter a multi-column tensor to a multi-column tensor
            if param_dims == 1:
                if conf.custom_scatter_cols:
                    return ops.scatter_cols(
                        params, indices,
                        pad_elem=tf.constant(0, dtype=params.dtype),
                        num_out_col=num_out_cols)
                else:
                    with_zeros = tf.concat(values=([0], params), axis=0)
                    gather_indices = np.zeros(num_out_cols, dtype=int)
                    gather_indices[indices] = np.arange(indices.size) + 1
                    return gather_cols(with_zeros, gather_indices)
            else:
                if conf.custom_scatter_cols:
                    return ops.scatter_cols(
                        params, indices,
                        pad_elem=tf.constant(0, dtype=params.dtype),
                        num_out_col=num_out_cols)
                else:
                    zero_col = tf.zeros((tf.shape(params)[0], 1),
                                        dtype=params.dtype)
                    with_zeros = tf.concat(values=(zero_col, params), axis=1)
                    gather_indices = np.zeros(num_out_cols, dtype=int)
                    gather_indices[indices] = np.arange(indices.size) + 1
                    return gather_cols(with_zeros, gather_indices)


def broadcast_value(value, shape, dtype, name=None):
    """Broadcast the given value to the given shape and dtype. If ``value`` is
    one of the members of :class:`~libspn.ValueType`, the requested value will
    be generated and placed in every element of a tensor of the requested shape
    and dtype. If ``value`` is a 0-D tensor or a Python value, it will be
    broadcasted to the requested shape and converted to the requested dtype.
    Otherwise, the value is used as is.

    Args:
        value: The input value.
        shape: The shape of the output.
        dtype: The type of the output.

    Return:
        Tensor: A tensor containing the broadcasted and converted value.
    """
    with tf.name_scope(name, "broadcast_value", [value]):
        # Recognize ValueTypes
        if isinstance(value, ValueType.RANDOM_UNIFORM):
            return tf.random_uniform(shape=shape,
                                     minval=value.min_val,
                                     maxval=value.max_val,
                                     dtype=dtype)

        # Broadcast tensors and scalars
        tensor = tf.convert_to_tensor(value, dtype=dtype)
        if tensor.get_shape() == tuple():
            return tf.fill(dims=shape, value=tensor)

        # Return original input if we cannot broadcast
        return tensor


def normalize_tensor(tensor, name=None):
    """Normalize the tensor so that all elements sum to 1.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    with tf.name_scope(name, "normalize_tensor", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        s = tf.reduce_sum(tensor)
        return tf.truediv(tensor, s)


def reduce_log_sum(log_input, name=None):
    """Calculate log of a sum of elements of a tensor containing log values
    row-wise.

    Args:
        log_input (Tensor): Tensor containing log values.

    Returns:
        Tensor: The reduced tensor of shape ``(None, 1)``, where the first
        dimension corresponds to the first dimension of ``log_input``.
    """
    with tf.name_scope(name, "reduce_log_sum", [log_input]):
        log_max = tf.reduce_max(log_input, -1, keepdims=True)
        # Compute the value assuming at least one input is not -inf
        log_rebased = tf.subtract(log_input, log_max)
        out_normal = log_max + tf.log(tf.reduce_sum(tf.exp(log_rebased),
                                                    -1, keepdims=True))
        # Check if all input values in a row are -inf (all non-log inputs are 0)
        # and produce output for that case
        # We use float('inf') for compatibility with Python<3.5
        # For Python>=3.5 we can use math.inf instead
        all_zero = tf.equal(log_max,
                            tf.constant(-float('inf'), dtype=log_input.dtype))
        out_zeros = tf.fill(tf.shape(out_normal),
                            tf.constant(-float('inf'), dtype=log_input.dtype))
        # Choose the output for each row
        return tf.where(all_zero, out_zeros, out_normal)


def concat_maybe(values, axis, name='concat'):
    """Concatenate ``values`` if there is more than one value. Oherwise, just
    forward value as is.

    Args:
        values (list of Tensor): Values to concatenate

    Returns:
        Tensor: Concatenated values.
    """
    if len(values) > 1:
        return tf.concat(values=values, axis=axis, name=name)
    else:
        return values[0]


def split_maybe(value, split_sizes, axis, name='split'):
    """Split ``value`` into multiple tensors of sizes given by ``split_sizes``.
    ``split_sizes`` must sum to the size of ``split_dim``. If only one split_size
    is given, the function does nothing and just forwards the value as the only
    split.

    Args:
        value (Tensor): The tensor to split.
        split_sizes (list of int): Sizes of each split.
        axis (int): The dimensions along which to split.

    Returns:
        list of Tensor: List of resulting tensors.
    """
    if len(split_sizes) > 1:
        return tf.split(value=value, num_or_size_splits=split_sizes,
                        axis=axis, name=name)
    else:
        return [value]


def print_tensor(*tensors):
    return tf.Print(tensors[0], tensors)
