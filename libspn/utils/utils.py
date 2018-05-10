# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
"""LibSPN tools and utilities."""
import functools

import libspn as spn


_MEMO_INDEX = 0


def decode_bytes_array(arr):
    """Convert an array of bytes objects to an array of Unicode strings."""
    if arr.dtype.hasobject and type(arr.item(0)) is bytes:
        return arr.astype(str)
    else:
        return arr


class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the memoize() will hash
        the key multiple times on a cache miss.

        Implementation taken from functools package
    """

    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(args, kwds, typed, kwd_mark=(object(),),
              fasttypes={int, str, frozenset, type(None)},
              sorted=sorted, tuple=tuple, type=type, len=len):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    Implementation taken from functools package
    """
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def memoize(f):
    return ArgumentCache.memoize(f)


class ArgumentCache:

    _CACHE_HELPERS = []

    @staticmethod
    def memoize(f):
        """ Allows for memoization that can be configured with spn.conf.memoization """

        class Helper(object):
            def __init__(self, func):
                self._memo = {}
                self._prev_memos = []
                self._skip_memos = []
                self.func = func

            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self.func
                return functools.partial(self, obj)

            def __call__(self, *args, **kwargs):
                if not spn.conf.memoization:
                    return f(*args, **kwargs)
                try:
                    key = _make_key(args, kwargs, typed=True)
                except TypeError as e:
                    print(e)
                    return f(*args, **kwargs)
                if key not in self._memo:
                    for i, m in enumerate(self._prev_memos):
                        if i in self._skip_memos:
                            continue
                        if key in m:
                            return m[key]
                    res = self._memo[key] = f(*args, **kwargs)
                    return res

                return self._memo[key]

            def set_ignore_memos(self, memos):
                self._skip_memos = memos

            def new_memo(self):
                self._prev_memos.append(self._memo)
                self._memo = {}

            def ignore_previous_memos(self):
                self._skip_memos = list(range(len(self._prev_memos)))

        helper = Helper(f)
        ArgumentCache._CACHE_HELPERS.append(helper)
        return helper

    @staticmethod
    def _increment_memos():
        [h.new_memo() for h in ArgumentCache._CACHE_HELPERS]

    @staticmethod
    def _ignore_previous_memos():
        [h.ignore_previous_memos() for h in ArgumentCache._CACHE_HELPERS]

    @staticmethod
    def _enable_all_memos():
        [h.set_ignore_memos([]) for h in ArgumentCache._CACHE_HELPERS]
