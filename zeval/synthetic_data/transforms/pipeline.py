"""Transformation pipeline"""

import asyncio
from typing import List
from rich.progress import Progress, TaskID
from .extractors.base import BaseExtractor
from ...schemas.base import BaseUnit


class TransformPipeline:
    """
    Transformation pipeline: concurrent + rate limiting
    
    Core logic:
    - Use Semaphore to control concurrency
    - tenacity handles retry
    - Skip on failure
    
    Attributes:
        extractors: List of extractors to run
        max_concurrency: Maximum concurrent LLM calls (default: 10)
    
    Example:
        pipeline = TransformPipeline(
            extractors=[summary_ext, keyphrases_ext],
            max_concurrency=10
        )
        enriched_units = await pipeline.transform(units)
    """
    
    def __init__(
        self,
        extractors: List[BaseExtractor],
        max_concurrency: int = 10,
    ):
        self.extractors = extractors
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def transform(self, units: List[BaseUnit]) -> List[BaseUnit]:
        """
        Transform all units concurrently
        
        Flow:
        1. Create all tasks (unit × extractor)
        2. Execute concurrently (limited by Semaphore)
        3. Extractors directly populate unit fields
        
        Args:
            units: List of document units to transform
            
        Returns:
            Units with populated fields (summary, keyphrases, entities)
        """
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
                    task = self._run_with_semaphore(unit, extractor, progress, task_id)
                    tasks.append(task)
            
            # Execute concurrently (extractors modify units in-place)
            await asyncio.gather(*tasks)
        
        return units
    
    async def _run_with_semaphore(
        self,
        unit: BaseUnit,
        extractor: BaseExtractor,
        progress: Progress,
        task_id: TaskID
    ) -> bool:
        """
        Run extraction with semaphore-based concurrency control
        
        Semaphore mechanism:
        - Acquire permit → execute → release permit
        - Max max_concurrency tasks run simultaneously
        
        Args:
            unit: Document unit
            extractor: Extractor instance
            progress: Rich progress instance
            task_id: Progress task ID
            
        Returns:
            True if successful, False if failed
        """
        async with self.semaphore:
            result = await extractor.extract(unit)
            progress.update(task_id, advance=1)
            return result
