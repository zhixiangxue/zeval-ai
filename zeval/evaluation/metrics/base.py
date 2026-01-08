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
        timeout: float = 120.0,
        show_progress: bool = True,
        progress_callback = None
    ) -> None:
        """
        Evaluate a batch of cases concurrently with optional progress display
        
        Results are written directly into case.results[self.name].
        
        Args:
            cases: List of evaluation cases
            concurrency: Maximum concurrent evaluations
            timeout: Timeout for each case evaluation in seconds (default: 120s)
            show_progress: Whether to show progress bar (default: True)
            progress_callback: Optional callback(completed, total) for external progress tracking
        """
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0
        total_count = len(cases)
        
        # Setup progress display
        if show_progress and not progress_callback:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.completed}/{task.total}"),
            )
            progress.start()
            task_id = progress.add_task(
                f"[cyan]Evaluating {self.name}",
                total=total_count
            )
        else:
            progress = None
            task_id = None
        
        try:
            async def eval_one(case: EvalCase, index: int):
                nonlocal completed_count
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
                        print(f"\n[WARNING] Case {index+1} timeout after {timeout}s")
                    except Exception as e:
                        # On failure, still record a result with score 0
                        case.add_result(EvalResult(
                            metric_name=self.name,
                            score=0.0,
                            reason=f"Evaluation failed: {str(e)}",
                            elapsed_time=time.time() - start_time,
                        ))
                        print(f"\n[ERROR] Case {index+1} failed: {e}")
                    finally:
                        completed_count += 1
                        
                        # Update progress
                        if progress and task_id is not None:
                            progress.update(task_id, completed=completed_count)
                        if progress_callback:
                            progress_callback(completed_count, total_count)
            
            # Evaluate all cases concurrently
            tasks = [eval_one(case, i) for i, case in enumerate(cases)]
            await asyncio.gather(*tasks)
        finally:
            if progress:
                progress.stop()
