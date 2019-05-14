from libspn.inference.type import InferenceType
from libspn.graph.basesum import BaseSum
import libspn.utils as utils
import tensorflow as tf
from libspn.exceptions import StructureError
from libspn.graph.weights import Weights
import numpy as np
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode
from libspn import conf
import abc

from libspn.utils.math import logconv_1x1, logmatmul


@utils.register_serializable
class SpatialSum(BaseSum, abc.ABC):
    """A container representing convolutional sums (which share the same input) in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this container.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this container.
        name (str): Name of the container.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this container that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, *values, num_channels=1, weights=None, ivs=None,
                 inference_type=InferenceType.MARGINAL, name="LocalSum",
                 grid_dim_sizes=None):

        self._channel_axis = 3
        self._num_channels = num_channels
        self._scope_mask = None
        try:
            self._grid_dim_sizes = values[0].output_shape_spatial[:2]
        except:
            self._grid_dim_sizes = grid_dim_sizes
        if isinstance(self._grid_dim_sizes, tuple):
            self._grid_dim_sizes = list(self._grid_dim_sizes)
        super().__init__(
            *values, num_sums=self._num_inner_sums(), weights=weights, latent_indicators=ivs,
            inference_type=inference_type, name=name, reduce_axis=4, op_axis=[1, 2])

    @abc.abstractmethod
    def _num_inner_sums(self):
        """Returns the number of sums that are modelled internally. """
        # TODO better terminology here

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _prepare_component_wise_processing(
            self, w_tensor, ivs_tensor, *input_tensors, zero_prob_val=0.0):
        w_tensor = tf.reshape(w_tensor, self._spatial_weight_shape())

        input_tensors = [self._spatial_reshape(t) for t in input_tensors]

        reducible_inputs = tf.concat(input_tensors, axis=self._reduce_axis)

        if ivs_tensor is not None:
            shape = [-1] + self._grid_dim_sizes + [self._num_channels, self._max_sum_size]
            ivs_tensor = tf.reshape(ivs_tensor, shape=shape)

        return w_tensor, ivs_tensor, reducible_inputs

    @abc.abstractmethod
    def _spatial_weight_shape(self):
        """Shape of weight tensor so that it can be applied spatially. """

    @property
    def output_shape_spatial(self):
        return tuple(self._grid_dim_sizes + [self._num_channels])

    def generate_weights(
        self, initializer=tf.initializers.random_uniform(), trainable=True, input_sizes=None,
        log=False, name=None):
        """Generate a weights node matching this sum node and connect it to
        this sum.

        The function calculates the number of weights based on the number
        of input values of this sum. Therefore, weights should be generated
        once all inputs are added to this node.

        Args:
            initializer: Initial value of the weights. For possible values, see
                :meth:`~libspn.utils.broadcast_value`.
            trainable (bool): See :class:`~libspn.Weights`.
            input_sizes (list of int): Pre-computed sizes of each input of
                this node.  If given, this function will not traverse the graph
                to discover the sizes.
            log (bool): If "True", the weights are represented in log space.
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
        num_values = max(self._sum_sizes)
        # Generate weights
        weights = Weights(
            initializer=initializer, num_weights=num_values, num_sums=self._num_sums,
            log=log, trainable=trainable, name=name)
        self.set_weights(weights)
        return weights

    def _spatial_reshape(self, t, forward=True):
        """Reshapes a Tensor ``t``` to one that represents the spatial dimensions.

        Args:
            t (Tensor): The ``Tensor`` to reshape.
            forward (bool): Whether to reshape for forward inference. If True, reshapes to
                ``[batch, rows, cols, 1, input_channels]``. Otherwise, reshapes to
                ``[batch, rows, cols, output_channels, input_channels]``.
        Returns:
             A reshaped ``Tensor``.
        """
        non_batch_dim_size = self._non_batch_dim_prod(t)
        if forward:
            input_channels = non_batch_dim_size // np.prod(self._grid_dim_sizes)
            return tf.reshape(t, [-1] + self._grid_dim_sizes + [1, input_channels])
        return tf.reshape(t, [-1] + self._grid_dim_sizes + [non_batch_dim_size // (
            self._max_sum_size * np.prod(self._grid_dim_sizes)), self._max_sum_size])

    def _non_batch_dim_prod(self, t):
        """Computes the product of the non-batch dimensions to be used for reshaping purposes.

        Args:
            t (Tensor): A ``Tensor`` for which to compute the product.

        Returns:
            An ``int``: product of non-batch dimensions.
        """
        non_batch_dim_size = np.prod([ds for i, ds in enumerate(t.shape.as_list())
                                      if i != self._batch_axis])
        return int(non_batch_dim_size)

    def _get_input_num_channels(self):
        """Returns a list of number of input channels for each value Input.

        Returns:
            A list of ints containing the number of channels.
        """
        _, _, *input_sizes = self.get_input_sizes()
        return [int(s // np.prod(self._grid_dim_sizes)) for s in input_sizes]

    @utils.docinherit(BaseSum)
    def _get_sum_sizes(self, num_sums):
        num_values = sum(self._get_input_num_channels())  # Skip ivs, weights
        return num_sums * int(np.prod(self._grid_dim_sizes)) * [num_values]

    @utils.docinherit(BaseSum)
    def _compute_out_size(self, *input_out_sizes):
        return int(np.prod(self._grid_dim_sizes) * self._num_channels)

    def _set_scope_mask(self, t):
        self._scope_mask = t

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _compute_log_value(self, w_tensor, ivs_tensor, *input_tensors,
                           dropconnect_keep_prob=None, matmul_or_conv=False,
                           noise=None, batch_noise=None):

        def maybe_add_noise(val):
            if noise is not None and noise != 0.0:
                with tf.name_scope("Noise"):
                    self.logger.debug1("{}: added noise {}".format(self, noise))
                    noise_tensor = tf.truncated_normal(
                        shape=tf.shape(val), stddev=noise, mean=0.0)
                    val += noise_tensor
            if batch_noise is not None and batch_noise != 0.0:
                if self._scope_mask is None:
                    raise StructureError("Should set scope mask externally")
                with tf.name_scope("BatchNoise"):
                    self.logger.debug1("{}: added batch noise {}".format(self, batch_noise))
                    shuffled = tf.stop_gradient(tf.manip.roll(val, shift=1, axis=0))
                    mask = tf.tile(tf.less(tf.expand_dims(self._scope_mask, axis=-1), batch_noise),
                                   [1, 1, 1, self._num_channels])
                    val = tf.where(mask, shuffled, val)
            return val

        dropconnect_keep_prob = utils.maybe_first(
            self._dropconnect_keep_prob, dropconnect_keep_prob)
        if matmul_or_conv and ivs_tensor is not None:
            self.logger.warn("Cannot use matmul when using latent indicators, setting matmul=False")
            matmul_or_conv = False
        else:
            matmul_or_conv = conf.dropout_mode != 'pairwise' \
                             or dropconnect_keep_prob is None or dropconnect_keep_prob == 1.0
        if matmul_or_conv:
            self.logger.debug1("{}: using matrix multiplication or conv ops.".format(self))
            w_tensor, _, inp_concat = self._prepare_component_wise_processing(
                w_tensor, ivs_tensor, *input_tensors)
            if dropconnect_keep_prob is not None and (not isinstance(
                    dropconnect_keep_prob, (int, float)) or dropconnect_keep_prob != 1.0):
                if conf.dropout_mode == "weights":
                    self.logger.debug1("{}: applying dropout with p={} to weights.".format(
                        self, dropconnect_keep_prob))
                    dropout_mask = self._create_dropconnect_mask(
                        keep_prob=dropconnect_keep_prob, shape=tf.shape(w_tensor))
                    min_inf = tf.cast(
                        tf.fill(tf.shape(w_tensor), value=float('-inf')), dtype=conf.dtype)
                    w_tensor = tf.where(dropout_mask, w_tensor, min_inf)
                    if conf.renormalize_dropconnect:
                        w_tensor = tf.nn.log_softmax(w_tensor, axis=-1)
                    if conf.rescale_dropconnect:
                        w_tensor -= tf.log(
                            dropconnect_keep_prob +
                            dropconnect_keep_prob ** w_tensor.shape[-1].value)
                else:  # Dropout to inputs (products)
                    self.logger.debug1("{}: applying dropout with p={} to sum inputs.".format(
                        self, dropconnect_keep_prob))
                    dropout_mask = self._create_dropconnect_mask(
                        keep_prob=dropconnect_keep_prob, shape=tf.shape(inp_concat))
                    min_inf = tf.cast(
                        tf.fill(tf.shape(inp_concat), value=float('-inf')), dtype=conf.dtype)
                    inp_concat = tf.where(dropout_mask, inp_concat, min_inf)
                    if conf.rescale_dropconnect:
                        inp_concat -= tf.log(
                            dropconnect_keep_prob +
                            dropconnect_keep_prob ** inp_concat.shape[-1].value)

            # Determine whether to use matmul or conv op and return
            if all(w_tensor.shape[i] == 1 for i in self._op_axis):
                w_tensor = tf.transpose(tf.squeeze(w_tensor, axis=0), (0, 1, 3, 2))
                inp_concat = tf.reshape(
                    inp_concat, [-1] + self._grid_dim_sizes + [self._max_sum_size])
                out = logconv_1x1(input=inp_concat, filter=w_tensor)
                return maybe_add_noise(tf.reshape(out, (-1, self._compute_out_size())))
            else:
                w_tensor = tf.squeeze(w_tensor, axis=0)
                inp_concat = tf.squeeze(inp_concat, axis=3)
                inp_concat = tf.reshape(
                    tf.transpose(inp_concat, (1, 2, 0, 3)),
                    self._grid_dim_sizes + [-1, self._max_sum_size])
                out = tf.transpose(logmatmul(inp_concat, w_tensor, transpose_b=True), (2, 0, 1, 3))
                out = maybe_add_noise(out)
                return tf.reshape(out, (-1, self._compute_out_size()))

        self.logger.debug1("{}: computing pairwise products.".format(self))
        val = self._reduce_marginal_inference_log(self._compute_reducible(
            w_tensor, ivs_tensor, *input_tensors, weighted=True,
            dropconnect_keep_prob=dropconnect_keep_prob))
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *input_tensors,
                           dropconnect_keep_prob=None):
        val = super(SpatialSum, self)._compute_log_mpe_value(
            w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=dropconnect_keep_prob)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_mpe_path_common(
            self, reducible_log_prob, counts, w_log_prob, latent_indicator_log_prob, *child_log_prob,
            sample=False, sample_prob=None, accumulate_weights_batch=False, use_unweighted=False):
        """Common operations for computing the MPE path.

        Args:
            reducible_log_prob (Tensor): A (weighted) ``Tensor`` of (log-)values of this container.
            counts (Tensor): A ``Tensor`` that contains the accumulated counts of the parents
                             of this container.
            w_log_prob (Tensor):  A ``Tensor`` containing the (log-)value of the weights.
            latent_indicator_log_prob (Tensor): A ``Tensor`` containing the logit of the
                latent indicators.
            child_log_prob (list): A list of ``Tensor``s with outputs of the child nodes.

        Returns:
            A ``list`` of ``tuple``s [(MPE counts, input tensor), ...] where the first corresponds
            to the Weights of this container, the second corresponds to the latent indicators and
            the remaining tuples correspond to the nodes in ``self._values``.
        """
        sample_prob = utils.maybe_first(sample_prob, self._sample_prob)
        num_samples = self._tile_unweighted_size if use_unweighted else 1
        if sample:
            max_indices = self._reduce_sample_log(
                reducible_log_prob, sample_prob=sample_prob, num_samples=num_samples)
        else:
            max_indices = self._reduce_argmax(reducible_log_prob, num_samples=num_samples)
        max_indices = tf.reshape(max_indices, (-1, self._compute_out_size()))
        max_counts = utils.scatter_values(
            params=counts, indices=max_indices, num_out_cols=self._max_sum_size)
        weight_counts, input_counts = self._accumulate_and_split_to_children(max_counts)

        if accumulate_weights_batch:
            weight_counts = tf.reduce_sum(weight_counts, axis=0, keepdims=False)
        return self._scatter_to_input_tensors(
            (weight_counts, w_log_prob),  # Weights
            (max_counts, latent_indicator_log_prob),  # Latent indicators
            *[(t, v) for t, v in zip(input_counts, child_log_prob)])  # Values

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _accumulate_and_split_to_children(self, x, *input_tensors):
        x = self._spatial_reshape(x, forward=False)
        x_acc_op = tf.reduce_sum(x, axis=self._op_axis)
        x_acc_channel_split = tf.split(
            tf.reduce_sum(x, axis=self._channel_axis),
            num_or_size_splits=self._get_input_num_channels(), axis=self._channel_axis)
        return x_acc_op, [self._flatten(t) for t in x_acc_channel_split]

    @utils.lru_cache
    def _flatten(self, t):
        """Flattens a Tensor ``t`` so that the resulting shape is [batch, non_batch]

        Args:
            t (Tensor): A ``Tensor```to flatten

        Returns:
            A flattened ``Tensor``.
        """
        if self._batch_axis != 0:
            raise NotImplementedError("{}: Cannot flatten if batch axis isn't equal to zero."
                                      .format(self))
        non_batch_dim_size = self._non_batch_dim_prod(t)
        return tf.reshape(t, (-1, non_batch_dim_size))

    @abc.abstractmethod
    def _accumulate_weight_counts(self, counts_spatial):
        """Accumulates the counts for the weights. """

    @utils.docinherit(BaseSum)
    def _compute_gradient(self, gradients, w_tensor, ivs_tensor, *input_tensors, with_ivs=True):
        raise NotImplementedError("{}: No gradient implementation available.".format(self))

    @utils.docinherit(BaseSum)
    def _compute_log_gradient(
            self, gradients, w_tensor, ivs_tensor, *value_tensors, with_ivs=True,
            accumulate_weights_batch=False, dropconnect_keep_prob=None):
        raise NotImplementedError("{}: No log-gradient implementation available.".format(self))

    @utils.docinherit(OpNode)
    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes, check_valid=False):
        value_scopes_grid = [
            np.asarray(vs).reshape(self._grid_dim_sizes + [-1]) for vs in value_scopes]
        value_scopes_concat = np.concatenate(value_scopes_grid, axis=2)

        if check_valid:
            for scope_list in value_scopes_concat.reshape((-1, self._max_sum_size)):
                if any(s != scope_list[0] for s in scope_list[1:]):
                    self.logger.warn("{}: not complete.".format(self))
                    return None

        if self._latent_indicators:
            raise NotImplementedError("{}: no support for computing scope when node has latent "
                                      "indicators.".format(self))
        return list(map(Scope.merge_scopes, value_scopes_concat.repeat(self._num_channels).reshape(
            (-1, self._max_sum_size))))

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes, check_valid=True)


    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _accumulate_and_split_to_children(self, x, *input_tensors):
        x = self._spatial_reshape(x, forward=False)
        x_acc_op = self._accumulate_weight_counts(x)
        x_acc_channel_split = tf.split(
            tf.reduce_sum(x, axis=self._channel_axis),
            num_or_size_splits=self._get_input_num_channels(), axis=self._channel_axis)
        return x_acc_op, [self._flatten(t) for t in x_acc_channel_split]
