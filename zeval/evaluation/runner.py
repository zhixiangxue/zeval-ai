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
    
    async def run(self, dataset: EvalDataset, show_progress: bool = True) -> EvalDataset:
        """
        Run all metrics on the dataset
        
        Args:
            dataset: Dataset to evaluate (will be modified in-place)
            show_progress: Whether to show progress bar
            
        Returns:
            The same dataset with evaluation results added
        """
        if show_progress:
            await self._run_with_progress(dataset)
        else:
            await self._run_without_progress(dataset)
        
        return dataset
    
    async def _run_with_progress(self, dataset: EvalDataset):
        """Run with rich progress bar showing all metrics"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        ) as progress:
            
            # Create progress tasks for all metrics
            metric_tasks = {}
            for metric in self.metrics:
                task_id = progress.add_task(
                    f"[cyan]{metric.name}", 
                    total=len(dataset.cases)
                )
                metric_tasks[metric.name] = task_id
            
            # Create evaluation tasks with progress callbacks
            async def eval_metric_with_progress(metric: BaseMetric):
                task_id = metric_tasks[metric.name]
                
                # Progress callback to update runner's progress
                def update_progress(completed: int, total: int):
                    progress.update(task_id, completed=completed)
                
                # Run metric with progress callback (disable metric's own progress)
                await metric.evaluate_batch(
                    dataset.cases,
                    show_progress=False,
                    progress_callback=update_progress
                )
            
            eval_tasks = [eval_metric_with_progress(m) for m in self.metrics]
            
            # Run all metrics concurrently
            await asyncio.gather(*eval_tasks)
    
    async def _run_without_progress(self, dataset: EvalDataset):
        """Run without progress display"""
        tasks = [metric.evaluate_batch(dataset.cases) for metric in self.metrics]
        await asyncio.gather(*tasks)
