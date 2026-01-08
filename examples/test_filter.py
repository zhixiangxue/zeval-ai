"""
Test dataset filtering with LLM-as-Judge

Demonstrates strict quality control on generated datasets.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich import print as rprint

from zeval.schemas.eval import EvalDataset
from zeval.synthetic_data.filters import GeneralFilter

load_dotenv()


async def main():
    """Test filtering on multi-hop dataset"""
    
    # Check for API key
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("Error: BAILIAN_API_KEY environment variable not set")
        return
    
    # Load the generated dataset
    dataset_path = "tmp/multi_hop_dataset.json"
    
    rprint(f"\n[bold cyan]Loading dataset from {dataset_path}...[/bold cyan]")
    dataset = EvalDataset.from_json(dataset_path)
    rprint(f"[green]✓[/green] Loaded {len(dataset.cases)} cases\n")
    
    # Create filter
    filter = GeneralFilter(
        uri="bailian/qwen-plus",  # Use qwen-plus for better judgment
        api_key=api_key,
        concurrency=3              # Don't overwhelm the API
    )
    
    # Apply strict filtering
    rprint("[bold cyan]Starting strict LLM-based filtering...[/bold cyan]")
    rprint("[yellow]Note: This will take a few minutes[/yellow]\n")
    
    filtered_dataset = await filter.filter(dataset)
    
    # Save filtered dataset
    output_path = "tmp/multi_hop_dataset_filtered.json"
    filtered_dataset.to_json(output_path)
    
    rprint(f"[green]✓[/green] Filtered dataset saved to {output_path}")
    
    # Print some filtered cases
    if filtered_dataset.cases:
        rprint(f"\n[bold]Sample of accepted cases:[/bold]")
        for i, case in enumerate(filtered_dataset.cases[:3], 1):
            rprint(f"\n[cyan]Case {i}:[/cyan]")
            rprint(f"Q: {case.question}")
            rprint(f"A: {case.ground_truth_answer[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
