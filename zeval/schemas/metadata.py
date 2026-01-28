"""
Metadata definitions for documents and units.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentMetadata(BaseModel):
    """
    Structured metadata for documents
    Type-safe alternative to plain dict
    """
    
    # Source information
    source: str
    source_type: str  # "local" or "url"
    file_type: str  # "pdf", "markdown", etc.
    
    # File information
    file_name: Optional[str] = None
    file_size: Optional[int] = None  # in bytes
    file_extension: Optional[str] = None
    
    # Content information
    content_length: int = 0  # length in characters
    mime_type: Optional[str] = None
    
    # Processing information
    created_at: datetime = Field(default_factory=datetime.now)
    reader_name: Optional[str] = None
    
    # Custom fields (for extensibility)
    custom: dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


class UnitMetadata(BaseModel):
    """
    Universal metadata for units
    Minimalist design inspired by LlamaIndex
    
    Focuses on providing context path for LLM understanding
    """
    
    context_path: Optional[str] = None
    """
    Hierarchical path providing full context
    
    This single field encodes:
    - Context hierarchy (by splitting "/")
    - Current section/page title (last segment)
    - Parent context (all but last segment)
    - Depth level (count "/")
    
    Examples:
        - Markdown: "Introduction/Background/History"
        - PDF: "Page3/Section2"
        - Word: "Chapter1/Section1.2"
        - Excel: "Sheet1/TableA"
    """
    
    custom: dict[str, Any] = Field(default_factory=dict)
    """
    Optional custom fields for document-type specific metadata
    
    Examples:
        - Markdown: {"is_code_block": True, "language": "python"}
        - PDF: {"page_number": 3, "bbox": [x, y, w, h]}
        - Table: {"row_count": 10, "col_count": 5}
    """
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
