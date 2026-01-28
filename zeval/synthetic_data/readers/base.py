"""
Base reader class for all readers
"""

import xxhash
from abc import ABC, abstractmethod
from pathlib import Path

from ...schemas.document import BaseDocument


class BaseReader(ABC):
    """
    Base class for all readers
    Simple interface definition - readers parse files and return Document objects
    
    Note:
        For optional utility functions (source validation, file type detection),
        see zag.utils.source.SourceUtils
    """
    
    @abstractmethod
    def read(self, source: str) -> BaseDocument:
        """
        Read and parse a file
        
        Args:
            source: File path (relative/absolute) or URL
            
        Returns:
            A Document object containing structured parsed results
        """
        pass
    
    @staticmethod
    def md5(file_path: str) -> str:
        """
        Compute file hash using xxHash (fast)
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash hex string (16 characters)
            
        Example:
            >>> from zeval.synthetic_data.readers.base import BaseReader
            >>> hash_value = BaseReader.md5("document.pdf")
            >>> print(hash_value)  # e.g., 'a3f5e1b2c4d6e8f0'
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file, 'rb') as f:
            return xxhash.xxh64(f.read()).hexdigest()
