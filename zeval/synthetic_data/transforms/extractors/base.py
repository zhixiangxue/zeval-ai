"""
Base extractor class for information extraction
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List
import chak
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from rich.progress import Progress
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
    
    def __or__(self, other: 'BaseExtractor') -> 'CompositeExtractor':
        """
        Compose extractors using | operator
        
        Example:
            combined = summary | entities | keyphrases
            await combined.transform(units)
        """
        if isinstance(self, CompositeExtractor):
            return CompositeExtractor(self.extractors + [other])
        return CompositeExtractor([self, other])


class CompositeExtractor:
    """
    Composite extractor that combines multiple extractors
    
    Supports pipeline-style composition with | operator:
        extractor = summary | entities | keyphrases
        await extractor.transform(units, max_concurrency=10)
    
    Attributes:
        extractors: List of extractors to run
    
    Example:
        combined = SummaryExtractor(...) | EntitiesExtractor(...)
        await combined.transform(units, max_concurrency=10)
    """
    
    def __init__(self, extractors: List[BaseExtractor]):
        self.extractors = extractors
    
    def __or__(self, other: BaseExtractor) -> 'CompositeExtractor':
        """Support chaining: a | b | c"""
        return CompositeExtractor(self.extractors + [other])
    
    async def transform(
        self,
        units: List[BaseUnit],
        max_concurrency: int = 10
    ) -> List[BaseUnit]:
        """
        Transform all units with all extractors concurrently
        
        Flow:
        1. Create all tasks (unit × extractor)
        2. Execute concurrently with semaphore-based rate limiting
        3. Extractors populate unit fields in-place
        
        Args:
            units: List of document units to transform
            max_concurrency: Maximum concurrent LLM calls
            
        Returns:
            Units with populated fields (summary, keyphrases, entities)
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        total_tasks = len(units) * len(self.extractors)
        
        with Progress() as progress:
            task_id = progress.add_task(
                f"[cyan]Processing {len(units)} units x {len(self.extractors)} extractors...",
                total=total_tasks
            )
            
            # Create all tasks
            tasks = []
            for unit in units:
                for extractor in self.extractors:
                    task = self._run_with_semaphore(
                        unit, extractor, semaphore, progress, task_id
                    )
                    tasks.append(task)
            
            # Execute concurrently
            await asyncio.gather(*tasks)
        
        return units
    
    async def _run_with_semaphore(
        self,
        unit: BaseUnit,
        extractor: BaseExtractor,
        semaphore: asyncio.Semaphore,
        progress: Progress,
        task_id
    ) -> bool:
        """Run extraction with semaphore-based concurrency control"""
        async with semaphore:
            result = await extractor.extract(unit)
            progress.update(task_id, advance=1)
            return result
