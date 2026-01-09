"""Metric runner for batch evaluation"""

import asyncio
from typing import List
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..schemas.eval import EvalDataset
from .metrics.base import BaseMetric


class MetricRunner:
    """
    Runner for executing multiple metrics on a dataset
    
    Executes metrics concurrently (by metric) where each metric
    processes all cases in batch.
    
    Usage:
        runner = MetricRunner(metrics=[faithfulness, context_recall])
        await runner.run(dataset)
        # dataset.cases now contain evaluation results
    """
    
    def __init__(self, metrics: List[BaseMetric]):
        """
        Initialize runner with metrics
        
        Args:
            metrics: List of metrics to execute
        """
        self.metrics = metrics
    
    async def run(self, dataset: EvalDataset) -> EvalDataset:
        """
        Run all metrics on the dataset
        
        Args:
            dataset: Dataset to evaluate (will be modified in-place)
            
        Returns:
            The same dataset with evaluation results added
        """
        # Run all metrics concurrently, each shows its own progress
        tasks = [metric.evaluate_batch(dataset.cases) for metric in self.metrics]
        await asyncio.gather(*tasks)
        
        return dataset
