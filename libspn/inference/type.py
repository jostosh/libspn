from enum import Enum


class InferenceType(Enum):
    """Types of inference."""
    MARGINAL = "marginal"
    """Use marginal inference."""

    MPE = "mpe"
    """Use MPE inference."""
