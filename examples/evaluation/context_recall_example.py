"""Test Context Recall metric"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.evaluation.metrics import ContextRecall

# Load environment variables
load_dotenv()

console = Console()


def create_test_cases() -> list[EvalCase]:
    """Create test cases for context recall evaluation"""
    
    cases = [
        # Case 1: Full recall - all information is in contexts
        EvalCase(
            question="What is the minimum down payment and credit score for FHA loans?",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5% and a credit score of at least 580.",
            retrieved_contexts=[
                "FHA loans allow first-time homebuyers to make down payments as low as 3.5%. These loans are backed by the Federal Housing Administration.",
                "To qualify for an FHA loan with the 3.5% down payment, borrowers need a credit score of at least 580. Lower scores may require higher down payments.",
            ],
            answer="",  # Not used in context recall
            ground_truth_contexts=[],  # Not used in context recall
        ),
        
        # Case 2: Partial recall - some information missing
        EvalCase(
            question="What are the requirements for FHA loans?",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%, a credit score of at least 580, and mortgage insurance premiums.",
            retrieved_contexts=[
                "FHA loans allow down payments as low as 3.5% for qualified borrowers.",
                "Borrowers typically need a credit score of 580 or higher for FHA loans.",
                # Missing: mortgage insurance information
            ],
            answer="",
            ground_truth_contexts=[],
        ),
        
        # Case 3: Low recall - most information missing
        EvalCase(
            question="What are the benefits of FHA loans?",
            ground_truth_answer="FHA loans offer low down payments, flexible credit requirements, competitive interest rates, and are assumable by future buyers.",
            retrieved_contexts=[
                "FHA loans are popular among first-time homebuyers due to their low down payment requirements.",
                # Missing: flexible credit, interest rates, assumable features
            ],
            answer="",
            ground_truth_contexts=[],
        ),
        
        # Case 4: Zero recall - completely wrong contexts
        EvalCase(
            question="What is the interest rate for FHA loans?",
            ground_truth_answer="FHA loan interest rates are typically competitive and vary based on market conditions, credit score, and down payment amount.",
            retrieved_contexts=[
                "Closing costs for home purchases typically range from 2-5% of the purchase price.",
                "Home insurance is required by most lenders and protects against property damage.",
            ],
            answer="",
            ground_truth_contexts=[],
        ),
        
        # Case 5: Perfect recall with detailed contexts
        EvalCase(
            question="How do I qualify for a mortgage pre-approval?",
            ground_truth_answer="To qualify for mortgage pre-approval, you need to provide proof of income, employment verification, credit report, and information about your debts and assets.",
            retrieved_contexts=[
                "Mortgage pre-approval requires submitting financial documents including recent pay stubs, W-2 forms, and tax returns to verify income.",
                "Lenders will check your employment history and contact your employer for verification during the pre-approval process.",
                "Your credit report will be pulled to assess your creditworthiness, and you'll need to disclose all debts including credit cards, student loans, and car payments.",
                "Asset documentation such as bank statements and investment accounts must be provided to show you have funds for down payment and closing costs.",
            ],
            answer="",
            ground_truth_contexts=[],
        ),
    ]
    
    return cases


async def main():
    """Main test function"""
    
    console.print("\n" + "="*60)
    console.print("Testing Context Recall Metric", style="bold cyan")
    console.print("="*60 + "\n")
    
    # Step 1: Create test cases
    console.print("[Step 1] Creating test cases...", style="yellow")
    cases = create_test_cases()
    dataset = EvalDataset(cases=cases)
    console.print(f"✓ Created {len(cases)} test cases\n", style="green")
    
    # Step 2: Initialize metric
    console.print("[Step 2] Initializing context recall metric...", style="yellow")
    llm_uri = "openai/gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key:
        console.print("✗ Error: OPENAI_API_KEY not found in environment", style="red")
        return
    
    metric = ContextRecall(llm_uri=llm_uri, api_key=api_key)
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
        result = case.results.get("context_recall")
        if not result:
            continue
        
        scores.append(result.score)
        
        # Determine recall level
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
        
        # Show retrieved contexts
        content.append(f"[bold]Retrieved Contexts:[/bold]")
        for j, ctx in enumerate(case.retrieved_contexts[:2], 1):
            preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
            content.append(f"  {j}. {preview}")
        if len(case.retrieved_contexts) > 2:
            content.append(f"  ... and {len(case.retrieved_contexts)-2} more contexts")
        content.append("")
        
        # Show recall score
        content.append(f"[bold]Recall Score:[/bold] [{level_style}]{result.score:.2f} ({level})[/{level_style}]\n")
        
        # Show reason
        content.append(f"[bold]Reason:[/bold]\n{result.reason}\n")
        
        # Show elapsed time
        if result.elapsed_time:
            content.append(f"[bold]Elapsed Time:[/bold] {result.elapsed_time:.2f}s\n")
        
        # Show statement breakdown if available
        if result.details and "classifications" in result.details:
            classifications = result.details["classifications"]
            content.append(f"[bold]Statement Breakdown:[/bold]")
            for cls in classifications[:3]:
                status = "✓" if cls["attributed"] == 1 else "✗"
                stmt_preview = cls["statement"][:80] + "..." if len(cls["statement"]) > 80 else cls["statement"]
                content.append(f"  {status} {stmt_preview}")
            if len(classifications) > 3:
                content.append(f"  ... and {len(classifications)-3} more statements")
        
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
        
        summary_table.add_row("Average Recall", f"{sum(scores)/len(scores):.2f}")
        summary_table.add_row("Max Recall", f"{max(scores):.2f}")
        summary_table.add_row("Min Recall", f"{min(scores):.2f}")
        summary_table.add_row("Total Cases", str(len(scores)))
        
        console.print(summary_table)
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
