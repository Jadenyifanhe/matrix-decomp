"""
Knowledge transfer utilities for moving decomposed weights between models.
"""

from .knowledge_transfer import KnowledgeTransfer
from .dimension_adapter import DimensionAdapter, adapt_dimensions

__all__ = [
    "KnowledgeTransfer",
    "DimensionAdapter",
    "adapt_dimensions",
]
