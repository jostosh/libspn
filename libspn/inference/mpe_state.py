# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.inference.mpe_path import MPEPath


class MPEState:
    """Assembles TF operations computing MPE state for an SPN.

    Args:
        mpe_path (MPEPath): Pre-computed MPE_path.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
    """

    def __init__(self, mpe_path=None, log=True, value_inference_type=None, dynamic=False):
        # Create internal MPE path generator
        self._dynamic = dynamic
        if mpe_path is None:
            self._mpe_path = MPEPath(log=log,
                                     value_inference_type=value_inference_type,
                                     add_random=None, use_unweighted=False, dynamic=dynamic,
                                     dynamic_reduce_in_loop=False, dropout_keep_prob=1.0, 
                                     dropconnect_keep_prob=1.0)
        else:
            self._mpe_path = mpe_path

    @property
    def mpe_path(self):
        """MPEPath: Computed MPE path."""
        return self._mpe_path

    def get_state(self, root, *var_nodes, sequence_lens=None):
        """Assemble TF operations computing the MPE state of the given SPN
        variables for the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
            *var_nodes (VarNode): Variable nodes for which the state should
                                  be computed.

        Returns:
            list of Tensor: A list of tensors containing the MPE state for the
            variable nodes.
        """
        # Generate path if not yet generated
        if not self._mpe_path.counts:
            self._mpe_path.get_mpe_path(root, sequence_lens=sequence_lens)

        with tf.name_scope("MPEState"):
            if self._dynamic:
                return tuple(var_node._compute_mpe_state(
                    self._mpe_path.counts_per_step[var_node]) for var_node in var_nodes)

            return tuple(var_node._compute_mpe_state(
                self._mpe_path.counts[var_node]) for var_node in var_nodes)
