# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.utils.serialization import register_serializable
import enum


class Enum(enum.Enum):
    """Serializable Enum."""

    def __init__(self, *args):
        # TODO: Move to metaclass
        register_serializable(type(self))

    def serialize(self):
        return {'value': self.name}

    @classmethod
    def deserialize(cls, data):
        """This function does not follow the standard deserialization pattern,
        but instade it just returns the value. The reason is that with enum, the
        value must be passed directly to __new__ and cannot be deserialized
        after __new__. Therefore, we use a workaround inside _decode_json for
        such case.
        """
        name = data['value']

        # Lookup the value of the enum based on name
        # enums are created using value not name
        return cls[name]
