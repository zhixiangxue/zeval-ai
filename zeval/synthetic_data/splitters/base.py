"""
Base splitter class for all splitters
"""

from abc import ABC, abstractmethod
from typing import Union
import uuid

from ...schemas.unit import BaseUnit, UnitCollection
from ...schemas.document import BaseDocument


class BaseSplitter(ABC):
    """
    Base class for all splitters
    Splitters can split documents or further split existing units
    """
    
    @abstractmethod
    def _do_split(self, input_data: Union[BaseDocument, list[BaseUnit]]) -> list[BaseUnit]:
        """
        Internal method to perform the actual splitting logic
        Subclasses should implement this method
        
        Args:
            input_data: Document or list of units to split
            
        Returns:
            List of units
        """
        pass
    
    def split(self, input_data: Union[BaseDocument, list[BaseUnit], UnitCollection]) -> UnitCollection:
        """
        Split document or units into smaller units
        
        Supports three input types:
        1. BaseDocument - Split document content
        2. list[BaseUnit] - Further split existing units
        3. UnitCollection - Extract and split units (UnitCollection is a list subclass)
        
        Args:
            input_data: Document, units list, or UnitCollection to split
            
        Returns:
            UnitCollection with units having chain relationships
        """
        # UnitCollection is a list subclass, no need to unwrap
        
        # Perform actual splitting
        units = self._do_split(input_data)
        
        # Set source_doc_id if input is a document
        if isinstance(input_data, BaseDocument):
            for unit in units:
                unit.source_doc_id = input_data.doc_id
        
        return UnitCollection(units)
    
    @staticmethod
    def generate_unit_id() -> str:
        """Generate a unique unit ID"""
        return f"unit_{uuid.uuid4().hex[:12]}"
