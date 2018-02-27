#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestUtils(TestCase):

    def test_memoize(self):
        @spn.utils.memoize
        def rand(dummy0, dummy1=None):
            return np.random.rand()
        spn.conf.memoization = True
        self.assertEqual(rand(1), rand(1))
        self.assertNotEqual(rand(1), rand(1, dummy1=False))

        spn.conf.memoization = False
        self.assertNotEqual(rand(1), rand(1))


if __name__ == '__main__':
    tf.test.main()
