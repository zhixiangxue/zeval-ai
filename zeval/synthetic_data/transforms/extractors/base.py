"""
Base extractor class for information extraction
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import chak
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from ....schemas.base import BaseUnit


class BaseExtractor(ABC):
    """
    Base class for all extractors
    
    Core logic:
    Input = model_uri + api_key
    Internal = creates Conversation instance
    Output = each extractor updates unit fields directly
    
    Attributes:
        model_uri: Model URI (e.g., "openai/gpt-4o-mini")
        api_key: API key for the model
    
    Example:
        extractor = SummaryExtractor(
            model_uri="openai/gpt-4o-mini",
            api_key="sk-xxx"
        )
        await extractor.extract(unit)  # populates unit.summary
    """
    
    def __init__(self, model_uri: str, api_key: str):
        self.model_uri = model_uri
        self.api_key = api_key
    
    def create_conv(self) -> chak.Conversation:
        """Create a new Conversation instance for each extraction"""
        return chak.Conversation(
            self.model_uri,
            api_key=self.api_key
        )
    
    @abstractmethod
    async def extract(self, unit: BaseUnit) -> bool:
        """
        Extract information and update unit
        
        Each extractor implements its own logic:
        1. Build prompt (however you want)
        2. Call self.conv.asend(prompt, schema=YourSchema)
        3. Update unit fields directly
        
        Use tenacity @retry decorator for retry logic.
        
        Args:
            unit: Document unit to extract from
            
        Returns:
            True if successful, False if failed
        """
        pass
