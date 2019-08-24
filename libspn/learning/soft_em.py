import tensorflow as tf
from libspn.inference.mpe_path import MPEPath
from libspn.inference.value import LogValue
from libspn.graph.weights import Weights


class SoftEMLearning:
    """Assembles TF operations performing EM learning of an SPN. 

    Args: 
        mpe_path (MPEPath): Pre-computed MPE_path. 
        value_inference_type (InferenceType): The inference type used during the 
            upwards pass through the SPN. Ignored if ``mpe_path`` is given. 
        log (bool): If ``True``, calculate the value in the log space. Ignored 
                    if ``mpe_path`` is given. 
    """

    def __init__(self, root, minimum_accumulator_multiplier=1e-4):
        self._root = root
        self._minimum_accumulator_multiplier = minimum_accumulator_multiplier
        self._val_gen = LogValue()
        self._root_val = self._val_gen.get_value(self._root)
        # Create a name scope 
        with tf.name_scope("SoftEMLearning") as self._name_scope:
            pass

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._val_gen

    def update_spn(self):
        # Generate all update operations 
        with tf.name_scope(self._name_scope):
            weight_nodes, weight_vars = zip(*[
                (n, n.variable) for n in self._val_gen.values.keys()
                if isinstance(n, Weights)
            ])

            w_grads = tf.gradients(self._root_val, list(weight_vars))

            accumulators_w = [
                tf.Variable(tf.ones_like(w) * self._minimum_accumulator_multiplier / tf.cast(tf.shape(w)[1], tf.float32))
                for w in weight_vars
            ]

            acc_update = [tf.assign_add(a, wg / (tf.reduce_sum(wg, axis=1, keepdims=True) + 1e-8))
                          for a, wg in zip(accumulators_w, w_grads)]

            with tf.control_dependencies(acc_update):
                accumulators_decayed = [
                    tf.maximum(acc - self._minimum_accumulator_multiplier / tf.cast(tf.shape(acc)[1], tf.float32), self._minimum_accumulator_multiplier / tf.cast(tf.shape(acc)[1], tf.float32))
                    for acc in accumulators_w]
                return tf.group(
                    *(
                        [tf.assign(w, acc / tf.reduce_sum(acc, axis=-1, keepdims=True))
                         for w, acc in zip(weight_vars, accumulators_decayed)]
                    )
                )

    def learn(self):
        """Assemble TF operations performing EM learning of the SPN."""
        return None