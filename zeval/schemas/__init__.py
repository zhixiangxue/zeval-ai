"""
Schema module - organized structure for documents, units, and metadata.
"""

# Core types
from .types import UnitType

# Metadata
from .metadata import DocumentMetadata, UnitMetadata

# Documents
from .document import BaseDocument, PageableDocument, Page

# Units
from .unit import BaseUnit, TextUnit, TableUnit, ImageUnit, UnitCollection

# Evaluations
from .eval import (
    EvalCase,
    EvalDataset,
    EvalResult,
)

__all__ = [
    # Types
    "UnitType",
    # Metadata
    "DocumentMetadata",
    "UnitMetadata",
    # Documents
    "BaseDocument",
    "PageableDocument",
    "Page",
    # Units
    "BaseUnit",
    "TextUnit",
    "TableUnit",
    "ImageUnit",
    "UnitCollection",
    # Evaluations
    "EvalCase",
    "EvalDataset",
    "EvalResult",
]