"""
Quick test: Filter functionality

Tests filtering on the generated multi-hop dataset.
"""

import os
import asyncio
from dotenv import load_dotenv
from rich import print as rprint

from zeval.schemas.eval import EvalDataset
from zeval.synthetic_data.filters import GeneralFilter

load_dotenv()


async def main():
    """Test filter on existing dataset"""
    
    # Check for API key
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("Error: BAILIAN_API_KEY environment variable not set")
        return
    
    rprint("\n[bold cyan]Testing Filter Functionality[/bold cyan]\n")
    
    # Load dataset
    dataset_path = "tmp/multi_hop_dataset.json"
    rprint(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
    
    try:
        dataset = EvalDataset.from_json(dataset_path)
        rprint(f"[green]✓[/green] Loaded {len(dataset.cases)} cases\n")
    except FileNotFoundError:
        rprint(f"[red]✗[/red] File not found: {dataset_path}")
        rprint("[yellow]Run test_multi_hop.py first to generate the dataset[/yellow]")
        return
    
    # Create filter
    rprint("[cyan]Creating filter (qwen-plus for judgment)...[/cyan]")
    filter = GeneralFilter(
        uri="bailian/qwen-plus",
        api_key=api_key,
        concurrency=3
    )
    rprint("[green]✓[/green] Filter created\n")
    
    # Apply filter
    rprint("[bold]Applying strict quality filter...[/bold]")
    rprint("[yellow]This may take a few minutes...[/yellow]\n")
    
    filtered_dataset = await filter.filter(dataset)
    
    # Summary
    original_count = len(dataset.cases)
    filtered_count = len(filtered_dataset.cases)
    rejected_count = original_count - filtered_count
    
    rprint(f"\n[bold]Summary:[/bold]")
    rprint(f"  Original: {original_count} cases")
    rprint(f"  [green]Accepted: {filtered_count} cases[/green]")
    rprint(f"  [red]Rejected: {rejected_count} cases[/red]")
    rprint(f"  Acceptance rate: {filtered_count/original_count*100:.1f}%\n")
    
    # Save filtered dataset
    output_path = "tmp/multi_hop_dataset_filtered.json"
    filtered_dataset.to_json(output_path)
    rprint(f"[green]✓[/green] Saved filtered dataset to {output_path}\n")
    
    rprint("[bold green]✓ Filter test completed![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
