"""
Base filter for dataset quality control
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel, Field
from rich import print as rprint

from ...schemas.eval import EvalCase, EvalDataset


class ValidationResult(BaseModel):
    """Validation result for a test case"""
    decision: str = Field(description="ACCEPT or REJECT")
    case: EvalCase = Field(description="The evaluated case")
    reason: str = Field(description="Brief explanation")
    failed_criteria: list[str] = Field(default_factory=list, description="List of failed criterion names")


@dataclass
class FilterReport:
    """Filtering report with statistics"""
    total_cases: int
    accepted_cases: int
    rejected_cases: int
    rejection_reasons: dict[str, int]
    
    def print_summary(self):
        """Print filtering summary"""
        rprint("\n[cyan]" + "="*60 + "[/cyan]")
        rprint("[bold cyan]Dataset Filtering Report[/bold cyan]")
        rprint("[cyan]" + "="*60 + "[/cyan]")
        
        rprint(f"\n[bold]Total Cases:[/bold] {self.total_cases}")
        rprint(f"[green]✓ Accepted:[/green] {self.accepted_cases} ({self.accepted_cases/self.total_cases*100:.1f}%)")
        rprint(f"[red]✗ Rejected:[/red] {self.rejected_cases} ({self.rejected_cases/self.total_cases*100:.1f}%)")
        
        if self.rejection_reasons:
            rprint("\n[bold]Rejection Reasons:[/bold]")
            for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                rprint(f"  • {reason}: {count} cases")
        
        rprint("[cyan]" + "="*60 + "[/cyan]\n")


class Filter(ABC):
    """
    Abstract base class for dataset filters
    
    Input: EvalDataset (raw)
    Output: EvalDataset (filtered)
    """
    
    @abstractmethod
    async def filter(self, dataset: EvalDataset) -> EvalDataset:
        """
        Filter dataset to improve quality (async)
        
        Args:
            dataset: Raw generated dataset
        
        Returns:
            Filtered dataset
        """
        pass
