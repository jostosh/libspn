import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger
from libspn.generation.spatial import ConvSPN
from libspn.examples.convspn.architecture import full_wicker

logger = get_logger()


class TestConvSPN(tf.test.TestCase):

    def test_wicker(self):
        rows = 16
        cols = 16
        num_vars = rows * cols
        num_vals = 2
        # spatial_dims = [rows, cols]
        ivs = spn.IndicatorLeaf(num_vars=num_vars, num_vals=num_vals)
        wicker_root, _ = full_wicker(ivs, [4, 4, 4, 4], [2, 2, 2, 2], edge_size=16,
                                     first_depthwise=False)
        spn.generate_weights(wicker_root, initializer=tf.initializers.random_uniform(0.0, 1.0))
        init = spn.initialize_weights(wicker_root)

        self.assertTrue(wicker_root.is_valid())
        log_val_op = tf.reduce_logsumexp(wicker_root.get_log_value())

        num_vars_to_vary = 4
        num_possibilities = num_vals ** num_vars_to_vary
        pow2 = num_vals ** np.arange(num_vars_to_vary)
        feed_corners = np.greater(np.logical_and(np.arange(num_possibilities).reshape(
            (num_possibilities, 1)), pow2.reshape((1, num_vars_to_vary))), 0)

        feed = np.ones((num_possibilities, num_vars)) * -1
        corner_indices = 0, cols - 1, (rows - 1) * cols, num_vars - 1

        feed[:, corner_indices] = feed_corners

        with self.test_session() as sess:
            sess.run(init)
            log_val_out, log_val = sess.run(
                [log_val_op, wicker_root.get_log_value()], feed_dict={ivs: feed})
        self.assertAllClose(log_val_out, 0.0)

    def test_wicker_scope(self):
        rows = 16
        cols = 16
        num_vars = rows * cols
        num_vals = 2
        spatial_dims = [rows, cols]
        ivs = spn.IndicatorLeaf(num_vars=num_vars, num_vals=num_vals)
        dense_generator = spn.DenseSPNGenerator(
            num_decomps=1, num_mixtures=4, num_subsets=2,
            node_type=spn.DenseSPNGenerator.NodeType.LAYER)
        convspn = ConvSPN()
        dense_heads = convspn.wicker_stack(
            ivs, stack_size=3, spatial_dims=spatial_dims, dense_generator=dense_generator,
            strides=(1, 2, 2), add_root=False)
        target_scope = spn.Scope.merge_scopes([spn.Scope(ivs, i) for i in range(num_vars)])
        for head in dense_heads:
            self.assertEqual(target_scope, head.get_scope()[0])
