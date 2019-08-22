import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from libspn import conf, utils
from libspn.graph.node import Node
from libspn.graph.leaf.continuous_base import _softplus_inverse_np
from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn.utils.initializers import Equidistant
from libspn.utils.serialization import register_serializable


@register_serializable
class NormalLeaf(LocationScaleLeaf):
    """A node representing multiple uni-variate Normal distributions for continuous input
    variables. Each variable will have `num_components` normal components. Each Normal
    component has its own location (mean) and scale (standard deviation). These parameters
    can be learned or fixed.

    Lack of evidence must be provided explicitly through
    feeding :py:attr:`~libspn.NormalLeaf.evidence`. By default, evidence is set to ``True``
    for all variables.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of random variables.
        num_components (int): Number of components per random variable.
        name (str): Name of the node
        initialization_data (numpy.ndarray): Numpy array containing the data for mean and variance
                                             initialization.
        estimate_scale (bool): Boolean marking whether to estimate variance from
                                       ``initialization_data``.
        loc_init (float or numpy.ndarray): If a float and there's no ``initialization_data``,
                                            all components are initialized with ``loc_init``. If
                                            an numpy.ndarray, must have shape
                                            ``[num_vars, num_components]``.
        scale_init (float): If a float and there's no ``initialization_data``, scales are
            initialized with ``scale_init``.
        trainable_loc (bool): Whether to make the location ``Variable`` trainable.
        trainable_scale (bool): Whether to make the scale ``Variable`` trainable.
        use_prior (bool): Use prior when initializing variances from data.
                          See :meth:`~libspn.NormalLeaf.initialize_from_quantiles`.
        prior_alpha (float): Alpha parameter for variance prior.
                             See :meth:`~libspn.NormalLeaf.initialize_from_quantiles`.
        prior_beta (float): Beta parameter for variance prior.
                             See :meth:`~libspn.NormalLeaf.initialize_from_quantiles`.
        min_scale(float): Minimum value for standard devation. Used for avoiding numerical
                            instabilities when computing (log) pdfs.
        evidence_indicator_feed (Tensor): Tensor feeding this node's evidence indicator. If
                                          ``None``, an internal placeholder with default value will
                                          be created.
    """

    def __init__(self, feed=None, evidence_indicator_feed=None, num_vars=1, num_components=2,
                 initialization_data=None, estimate_scale=False, total_counts_init=1.0,
                 trainable_loc=True, trainable_scale=False,
                 loc_init=Equidistant(), scale_init=1.0,
                 min_scale=1e-2, softplus_scale=True, name="NormalLeaf",
                 share_locs_across_vars=False, share_scales=False, samplewise_normalization=False):
        self._estimate_scale = estimate_scale
        super().__init__(
            feed=feed, name=name, dimensionality=1, num_components=num_components,
            num_vars=num_vars, evidence_indicator_feed=evidence_indicator_feed,
            softplus_scale=softplus_scale, loc_init=loc_init, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, scale_init=scale_init, min_scale=min_scale,
            total_counts_init=total_counts_init, share_locs_across_vars=share_locs_across_vars,
            share_scales=share_scales, samplewise_normalization=samplewise_normalization,
            initialization_data=initialization_data)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.Normal(self._loc_variable, tf.nn.softplus(self._scale_variable))
        return tfd.Normal(self._loc_variable, self._scale_variable)

    def initialize_from_quantiles(self, data, num_quantiles):
        """Initializes the data from its quantiles per variable using the method described in
        Poon&Domingos UAI'11.

        Args:
            data (numpy.ndarray): Numpy array of shape [batch, num_vars] containing the data to
            initialize the means and variances.
            estimate_variance (bool): Whether to use the variance estimate.
            use_prior (False):  If ``True``, puts an inverse Gamma prior on the variance with
                                parameters ``prior_beta`` and ``prior_alpha``.
            prior_alpha (float): The alpha parameter of the inverse Gamma prior.
            prior_beta (float): The beta parameter of the inverse Gamma prior.
        """
        values_per_quantile = self._split_in_quantiles(data, num_quantiles)

        means = [np.mean(values, axis=0) for values in values_per_quantile]
        self._loc_init = np.stack(means, axis=-1)

        if self._estimate_scale:
            scales = [np.std(values, axis=0) for values in values_per_quantile]
            scales = np.stack(scales, axis=-1)
            self._scale_init = _softplus_inverse_np(scales) if self._softplus_scale else scales
            self._scale_init = np.maximum(self._scale_init, self._min_scale)

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        # Determine accumulates per component
        accum = tf.reduce_sum(counts_reshaped, axis=0)

        # Tile the feed
        tiled_feed = self._preprocessed_feed()
        data_per_component = tf.multiply(counts_reshaped, tiled_feed, name="DataPerComponent")
        squared_data_per_component = tf.multiply(
            counts_reshaped, tf.square(tiled_feed), name="SquaredDataPerComponent")
        sum_data = tf.reduce_sum(data_per_component, axis=0)
        sum_data_squared = tf.reduce_sum(squared_data_per_component, axis=0)
        return {'accum': accum, "sum_data": sum_data, "sum_data_squared": sum_data_squared}

    def assign(self, accum, sum_data, sum_data_squared):
        """
        Assigns new values to variables based on accumulated tensors. It updates the distribution
        parameters based on what can be found in "Online Structure Learning for Sum-Product Networks
        with Gaussian Leaves" by Hsu et al. (2017) https://arxiv.org/pdf/1701.05265.pdf

        Args:
            accum (Tensor): A ``Variable`` with accumulated counts per component.
            sum_data (Tensor): A ``Variable`` with the accumulated sum of data per component.
            sum_data_squared (Tensor): A ``Variable`` with the accumulated sum of squares of data
                                       per component.
        Returns:
            Tuple: A tuple containing assignment operations for the new total counts, the variance
            and the mean.
        """
        n = tf.maximum(self._total_count_variable, tf.ones_like(self._total_count_variable))
        k = accum
        mean = self._compute_hard_em_mean(k, n, sum_data)

        current_var = tf.square(self.scale_variable) if not self._softplus_scale else \
            tf.square(tf.nn.softplus(self._scale_variable))
        variance = (n * current_var + sum_data_squared -
                    2 * self.loc_variable * sum_data + k * tf.square(self.loc_variable)) / \
                   (n + k) - tf.square(mean - self._loc_variable)

        scale = tf.sqrt(variance)
        if self._softplus_scale:
            scale = tfd.softplus_inverse(scale)
        scale = tf.maximum(scale, self._min_scale)
        
        with tf.control_dependencies([n, mean, scale]):
            return (
                tf.assign_add(self._total_count_variable, k),
                tf.assign(self._scale_variable, scale) if self._trainable_scale else tf.no_op(),
                tf.assign(self._loc_variable, mean) if self._trainable_loc else tf.no_op())

    def assign_add(self, delta_loc, delta_scale):
        """
        Updates distribution parameters by adding a small delta value.

        Args:
            delta_loc (Tensor): A delta ``Tensor`` for the locations of the distributions.
            delta_scale (Tensor): A delta ``Tensor`` for the scales of the distributions.
        Returns:
             Tuple: An update ``Op`` for the locations and an update ``Op`` for the scales.
        """
        new_var = tf.maximum(self._scale_variable + delta_scale, self._min_scale)
        with tf.control_dependencies([new_var]):
            update_variance = tf.assign(self._scale_variable, new_var)
        return tf.assign_add(self._loc_variable, delta_loc), update_variance
