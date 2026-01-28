"""Test EvaluationReporter - Generate analysis report from evaluated dataset"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console

from zeval.evaluation.reporter import EvaluationReporter

# Load environment variables
load_dotenv()

console = Console()


async def main():
    """Main test function"""
    
    console.print("\n" + "="*70)
    console.print("Testing EvaluationReporter", style="bold cyan")
    console.print("="*70 + "\n")
    
    # Step 1: Initialize reporter
    console.print("[Step 1] Initializing reporter...", style="yellow")
    llm_uri = "openai/gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key:
        console.print("✗ Error: OPENAI_API_KEY not found in environment", style="red")
        return
    
    reporter = EvaluationReporter(llm_uri=llm_uri, api_key=api_key)
    console.print("✓ Reporter initialized\n", style="green")
    
    # Step 2: Load evaluated dataset
    console.print("[Step 2] Loading evaluated dataset...", style="yellow")
    dataset_path = "output/evaluated_dataset.json"
    
    # Check if file exists
    from pathlib import Path
    if not Path(dataset_path).exists():
        console.print(f"✗ Error: Dataset file not found: {dataset_path}", style="red")
        console.print("  Please run test_runner_all_metrics.py first to generate the dataset", style="yellow")
        return
    
    console.print(f"✓ Found dataset at {dataset_path}\n", style="green")
    
    # Step 3: Generate report
    console.print("[Step 3] Generating report (this may take a minute)...\n", style="yellow")
    
    output_path = "output/evaluation_report.md"
    
    report = await reporter.generate_report(
        dataset_path=dataset_path,
        output_path=output_path
    )
    
    console.print(f"\n✓ Report generated successfully\n", style="green")
    
    # Step 4: Display report preview
    console.print("="*70)
    console.print("Report Preview (first 2000 characters)", style="bold cyan")
    console.print("="*70 + "\n")
    
    preview = report[:2000]
    console.print(preview)
    
    if len(report) > 2000:
        console.print("\n...")
        console.print(f"\n[Full report has {len(report)} characters]\n", style="dim")
    
    console.print("="*70)
    console.print("✓ Report successfully generated", style="bold green")
    console.print("  Check the timestamped folder in output/ directory", style="dim")
    console.print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
