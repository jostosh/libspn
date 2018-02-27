# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""Global configuration options of LibSPN."""

import tensorflow as tf

dtype = tf.float32
"""Default dtype used by LibSPN."""

custom_gather_cols = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.gather_cols`."""

custom_scatter_cols = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.scatter_cols`."""

memoization = True
"""Whether to use LRU caches to function
return values in successive calls for reduced
graph size."""
