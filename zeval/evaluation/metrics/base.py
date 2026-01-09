"""Base metric class for evaluation"""

from abc import ABC, abstractmethod
import time
import asyncio
from typing import List
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskProgressColumn, BarColumn

from ...schemas.eval import EvalCase, EvalResult


class Statement(BaseModel):
    """A single atomic statement extracted from text"""
    text: str = Field(description="Statement text")
    supported: bool = Field(description="Whether supported by context")


class BaseMetric(ABC):
    """
    Base class for all evaluation metrics
    
    Each metric evaluates a batch of cases and writes results directly
    into case.results[metric_name].
    
    Subclasses should implement:
    - _evaluate(): Evaluate a single case and return EvalResult
    """
    
    def __init__(self, name: str):
        """
        Initialize metric
        
        Args:
            name: Metric name (e.g., 'faithfulness', 'context_recall')
        """
        self.name = name
    
    @abstractmethod
    async def _evaluate(self, case: EvalCase) -> EvalResult:
        """
        Evaluate a single case (to be implemented by subclasses)
        
        Args:
            case: Evaluation case to assess
            
        Returns:
            EvalResult with metric_name, score, reason, and optional details
        """
        pass
    
    async def evaluate_batch(
        self, 
        cases: List[EvalCase], 
        concurrency: int = 10, 
        timeout: float = 120.0
    ) -> None:
        """
        Evaluate a batch of cases concurrently with progress display
        
        Results are written directly into case.results[self.name].
        
        Args:
            cases: List of evaluation cases
            concurrency: Maximum concurrent evaluations
            timeout: Timeout for each case evaluation in seconds (default: 120s)
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        # Setup progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task_id = progress.add_task(
                f"Evaluating {self.name}",
                total=len(cases)
            )
            
            async def eval_one(case: EvalCase, index: int):
                async with semaphore:
                    start_time = time.time()
                    try:
                        # Add timeout wrapper
                        result = await asyncio.wait_for(
                            self._evaluate(case),
                            timeout=timeout
                        )
                        result.elapsed_time = time.time() - start_time
                        case.add_result(result)
                    except asyncio.TimeoutError:
                        # Handle timeout
                        case.add_result(EvalResult(
                            metric_name=self.name,
                            score=0.0,
                            reason=f"Evaluation timeout after {timeout}s",
                            elapsed_time=timeout,
                        ))
                    except Exception as e:
                        # On failure, still record a result with score 0
                        case.add_result(EvalResult(
                            metric_name=self.name,
                            score=0.0,
                            reason=f"Evaluation failed: {str(e)}",
                            elapsed_time=time.time() - start_time,
                        ))
                    finally:
                        # Update progress
                        progress.update(task_id, advance=1)
            
            # Evaluate all cases concurrently
            tasks = [eval_one(case, i) for i, case in enumerate(cases)]
            await asyncio.gather(*tasks)
