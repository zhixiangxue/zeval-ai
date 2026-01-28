"""
Document base classes and page structures.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid

from .metadata import DocumentMetadata


class Page(BaseModel):
    """
    Generic page structure for documents with page-level data
    """
    
    page_number: int
    content: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


class BaseDocument(BaseModel, ABC):
    """
    Base class for all document types
    Acts as a container for structured parsed results
    
    Note:
        doc_id is automatically generated if not provided
        metadata can be None and will be created on demand
    """
    
    doc_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    metadata: Optional[DocumentMetadata] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }
    
    @abstractmethod
    def split(self, splitter: 'BaseSplitter') -> 'UnitCollection':
        """
        Split document into units using the given splitter
        
        Args:
            splitter: The splitter to use for splitting
            
        Returns:
            UnitCollection containing the split units
        """
        pass


class PageableDocument(BaseDocument):
    """
    Base class for documents with page structure (PDF, DOCX, PPTX, etc.)
    """
    
    pages: list[Page] = Field(default_factory=list)
    
    def get_page(self, page_num: int) -> Optional[Page]:
        """
        Get page by page number
        
        Args:
            page_num: Page number to retrieve
            
        Returns:
            Page object if found, None otherwise
        """
        for page in self.pages:
            if page.page_number == page_num:
                return page
        return None
    
    def get_page_count(self) -> int:
        """
        Get total number of pages
        
        Returns:
            Total page count
        """
        return len(self.pages)
