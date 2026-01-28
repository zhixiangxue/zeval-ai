"""Test Context Precision metric"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.evaluation.metrics import ContextPrecision

# Load environment variables
load_dotenv()

console = Console()


def create_test_cases() -> list[EvalCase]:
    """Create test cases for context precision evaluation"""
    
    cases = [
        # Case 1: Perfect precision - all relevant contexts ranked at top
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%.",
            retrieved_contexts=[
                "FHA loans allow first-time homebuyers to make down payments as low as 3.5%. These loans are backed by the Federal Housing Administration.",  # Relevant
                "To qualify for an FHA loan with the 3.5% down payment, borrowers need a credit score of at least 580.",  # Relevant
                "Conventional loans typically require 5-10% down payment for most borrowers.",  # Less relevant
            ],
            answer="",  # Not used in context precision
            ground_truth_contexts=[],  # Not used in this metric
        ),
        
        # Case 2: Low precision - relevant context ranked last
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%.",
            retrieved_contexts=[
                "Closing costs typically range from 2-5% of the home purchase price.",  # Not relevant
                "Home insurance protects against property damage and is required by lenders.",  # Not relevant
                "FHA loans allow down payments as low as 3.5% for qualified borrowers.",  # Relevant (but ranked last - bad!)
            ],
            answer="",
            ground_truth_contexts=[],
        ),
        
        # Case 3: Medium precision - mixed ranking
        EvalCase(
            question="What are the benefits of fixed-rate mortgages?",
            ground_truth_answer="Fixed-rate mortgages offer predictable monthly payments and protection against rising interest rates.",
            retrieved_contexts=[
                "Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments.",  # Relevant
                "Adjustable-rate mortgages start with lower rates that adjust periodically.",  # Not relevant
                "Fixed-rate mortgages protect borrowers from rising interest rates by locking in the rate.",  # Relevant
                "Home equity lines of credit have variable interest rates.",  # Not relevant
            ],
            answer="",
            ground_truth_contexts=[],
        ),
        
        # Case 4: Zero precision - no relevant contexts
        EvalCase(
            question="What is the interest rate for FHA loans?",
            ground_truth_answer="FHA loan interest rates vary based on market conditions, credit score, and down payment amount.",
            retrieved_contexts=[
                "Property taxes are assessed by local governments based on home value.",
                "Homeowners insurance covers damage from fire, theft, and natural disasters.",
                "HOA fees vary by neighborhood and cover community amenities.",
            ],
            answer="",
            ground_truth_contexts=[],
        ),
        
        # Case 5: High precision - mostly relevant, well-ranked
        EvalCase(
            question="How do I qualify for a mortgage pre-approval?",
            ground_truth_answer="You need to provide proof of income, employment verification, credit report, and asset information.",
            retrieved_contexts=[
                "Mortgage pre-approval requires submitting pay stubs, W-2 forms, and tax returns to verify income.",  # Relevant
                "Lenders check your employment history and contact your employer during pre-approval.",  # Relevant
                "Credit reports are pulled to assess your creditworthiness for pre-approval.",  # Relevant
                "Bank statements show you have funds for down payment and closing costs.",  # Relevant
                "Real estate agents can recommend good neighborhoods for first-time buyers.",  # Not relevant
            ],
            answer="",
            ground_truth_contexts=[],
        ),
    ]
    
    return cases


async def main():
    """Main test function"""
    
    console.print("\n" + "="*60)
    console.print("Testing Context Precision Metric", style="bold cyan")
    console.print("="*60 + "\n")
    
    # Step 1: Create test cases
    console.print("[Step 1] Creating test cases...", style="yellow")
    cases = create_test_cases()
    dataset = EvalDataset(cases=cases)
    console.print(f"✓ Created {len(cases)} test cases\n", style="green")
    
    # Step 2: Initialize metric
    console.print("[Step 2] Initializing context precision metric...", style="yellow")
    llm_uri = "openai/gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key:
        console.print("✗ Error: OPENAI_API_KEY not found in environment", style="red")
        return
    
    metric = ContextPrecision(llm_uri=llm_uri, api_key=api_key)
    console.print("✓ Metric initialized\n", style="green")
    
    # Step 3: Evaluate cases
    console.print("[Step 3] Evaluating cases...", style="yellow")
    await metric.evaluate_batch(dataset.cases, concurrency=3)
    console.print(f"✓ Evaluated {len(cases)} cases\n", style="green")
    
    # Step 4: Display results
    console.print("="*60)
    console.print("Evaluation Results", style="bold cyan")
    console.print("="*60 + "\n")
    
    scores = []
    
    for i, case in enumerate(dataset.cases, 1):
        result = case.results.get("context_precision")
        if not result:
            continue
        
        scores.append(result.score)
        
        # Determine precision level
        if result.score >= 0.8:
            level = "High"
            level_style = "green"
        elif result.score >= 0.5:
            level = "Medium"
            level_style = "yellow"
        else:
            level = "Low"
            level_style = "red"
        
        # Create panel for each case
        content = []
        content.append(f"[bold]Question:[/bold]\n{case.question}\n")
        
        # Show ground truth answer
        content.append(f"[bold]Ground Truth Answer:[/bold]\n{case.ground_truth_answer}\n")
        
        # Show retrieved contexts with verdicts
        content.append(f"[bold]Retrieved Contexts (with verdicts):[/bold]")
        if result.details and "verdict_details" in result.details:
            verdict_details = result.details["verdict_details"]
            for detail in verdict_details[:5]:  # Show first 5
                status = "✓" if detail["verdict"] == 1 else "✗"
                status_color = "green" if detail["verdict"] == 1 else "red"
                content.append(f"  [{status_color}]{status}[/{status_color}] [{detail['position']}] {detail['context_preview']}")
            if len(verdict_details) > 5:
                content.append(f"  ... and {len(verdict_details)-5} more contexts")
        else:
            for j, ctx in enumerate(case.retrieved_contexts[:3], 1):
                preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
                content.append(f"  [{j}] {preview}")
        content.append("")
        
        # Show precision score
        content.append(f"[bold]Precision Score:[/bold] [{level_style}]{result.score:.2f} ({level})[/{level_style}]\n")
        
        # Show useful count
        if result.details:
            useful = result.details.get("useful_count", 0)
            total = result.details.get("total_count", 0)
            content.append(f"[bold]Useful Contexts:[/bold] {useful}/{total}\n")
        
        # Show reason
        content.append(f"[bold]Reason:[/bold]\n{result.reason}\n")
        
        # Show elapsed time
        if result.elapsed_time:
            content.append(f"[bold]Elapsed Time:[/bold] {result.elapsed_time:.2f}s")
        
        panel = Panel(
            "\n".join(content),
            title=f"[bold]Case {i}[/bold]",
            border_style="cyan"
        )
        console.print(panel)
        console.print()
    
    # Summary statistics
    if scores:
        summary_table = Table(title="📊 Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Average Precision", f"{sum(scores)/len(scores):.2f}")
        summary_table.add_row("Max Precision", f"{max(scores):.2f}")
        summary_table.add_row("Min Precision", f"{min(scores):.2f}")
        summary_table.add_row("Total Cases", str(len(scores)))
        
        console.print(summary_table)
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
