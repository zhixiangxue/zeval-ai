"""
Unit class definitions.
"""

from typing import Any, Optional, Callable
from pydantic import BaseModel, Field

from .types import UnitType
from .metadata import UnitMetadata


class BaseUnit(BaseModel):
    """
    Base class for all units (text, table, image, etc.)
    Simplified version - removed complex relationship management
    """
    
    unit_id: str
    content: Any
    unit_type: UnitType = UnitType.BASE
    metadata: Optional[UnitMetadata] = None
    
    # Embedding vector (optional, for caching)
    embedding: Optional[list[float]] = None
    
    # Retrieval score (set by retrievers)
    score: Optional[float] = None
    
    # Chain relationships (managed by Splitter)
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None
    source_doc_id: Optional[str] = None
    
    # Extracted properties (populated by extractors)
    summary: Optional[str] = None
    keyphrases: Optional[list[str]] = None
    entities: Optional[list[str]] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }


class TextUnit(BaseUnit):
    """Unit representing text content"""
    
    unit_type: UnitType = UnitType.TEXT
    content: str = ""


class TableUnit(BaseUnit):
    """Unit representing table content"""
    
    unit_type: UnitType = UnitType.TABLE
    content: str = ""  # Markdown/HTML/text format
    json_data: Optional[dict] = None  # Structured table data
    caption: Optional[str] = None


class ImageUnit(BaseUnit):
    """Unit representing image content"""
    
    unit_type: UnitType = UnitType.IMAGE
    content: bytes = b""
    format: Optional[str] = None  # "png", "jpg", "webp", etc.
    caption: Optional[str] = None


class UnitCollection(list):
    """
    Collection of units with chainable methods
    Supports filtering, extraction, and other operations
    """
    
    def __init__(self, units: list[BaseUnit]):
        super().__init__(units)
    
    def extract(self, extractor: 'BaseExtractor') -> 'UnitCollection':
        """
        Apply extractor to all units
        
        Args:
            extractor: The extractor to apply
            
        Returns:
            New UnitCollection with processed units
        """
        return UnitCollection([extractor.process(unit) for unit in self])
    
    def filter_by_type(self, unit_type: str) -> 'UnitCollection':
        """
        Filter units by type
        
        Args:
            unit_type: The unit type to filter ("text", "table", "image")
            
        Returns:
            New UnitCollection with filtered units
        """
        return UnitCollection([u for u in self if u.unit_type == unit_type])
    
    def filter(self, predicate: Callable[[BaseUnit], bool]) -> 'UnitCollection':
        """
        Filter units by custom predicate
        
        Args:
            predicate: A function that takes a unit and returns bool
            
        Returns:
            New UnitCollection with filtered units
        """
        return UnitCollection([u for u in self if predicate(u)])
    
    def get_by_id(self, unit_id: str) -> Optional[BaseUnit]:
        """
        Get unit by ID from this collection
        
        Args:
            unit_id: The unit ID to search for
            
        Returns:
            The unit if found, None otherwise
        """
        for unit in self:
            if unit.unit_id == unit_id:
                return unit
        return None
    
    def get_text_units(self) -> 'UnitCollection':
        """Shortcut to get all text units"""
        return self.filter_by_type("text")
    
    def get_table_units(self) -> 'UnitCollection':
        """Shortcut to get all table units"""
        return self.filter_by_type("table")
    
    def get_image_units(self) -> 'UnitCollection':
        """Shortcut to get all image units"""
        return self.filter_by_type("image")

