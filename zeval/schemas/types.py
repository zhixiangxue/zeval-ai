"""
Type definitions and enums for schema types.
"""

from enum import Enum


class UnitType(str, Enum):
    """Unit content types"""
    
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    BASE = "base"  # Base/unknown type
