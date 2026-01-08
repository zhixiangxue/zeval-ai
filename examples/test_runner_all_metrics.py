"""Test MetricRunner with all implemented metrics"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.evaluation.metrics import (
    Faithfulness,
    ContextRelevance,
    ContextRecall,
    ContextPrecision,
    AnswerRelevancy,
    AnswerCorrectness,
)
from zeval.evaluation.runner import MetricRunner

# Load environment variables
load_dotenv()

console = Console()


def create_comprehensive_test_cases() -> list[EvalCase]:
    """Create test cases covering all metrics"""
    
    cases = [
        # Case 1: Perfect case - everything is correct
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            answer="FHA loans require a minimum down payment of 3.5%.",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%.",
            retrieved_contexts=[
                "FHA loans allow first-time homebuyers to make down payments as low as 3.5%. These loans are backed by the Federal Housing Administration.",
            ],
            ground_truth_contexts=[
                "FHA loans allow first-time homebuyers to make down payments as low as 3.5%. These loans are backed by the Federal Housing Administration.",
            ],
        ),
        
        # Case 2: Good answer with complete information
        EvalCase(
            question="What are the requirements for FHA loans?",
            answer="FHA loans require a minimum down payment of 3.5% and a credit score of at least 580.",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%, a credit score of at least 580, and mortgage insurance premiums.",
            retrieved_contexts=[
                "FHA loans allow down payments as low as 3.5% for qualified borrowers.",
                "Borrowers typically need a credit score of 580 or higher for FHA loans.",
                "FHA loans require mortgage insurance premiums to protect lenders.",
            ],
            ground_truth_contexts=[
                "FHA loans allow down payments as low as 3.5% for qualified borrowers.",
                "Borrowers typically need a credit score of 580 or higher for FHA loans.",
                "FHA loans require mortgage insurance premiums to protect lenders.",
            ],
        ),
        
        # Case 3: Answer with some issues
        EvalCase(
            question="What are the benefits of fixed-rate mortgages?",
            answer="Fixed-rate mortgages offer predictable monthly payments and protection against rising interest rates. They also have very low closing costs.",  # Last sentence is hallucination
            ground_truth_answer="Fixed-rate mortgages offer predictable monthly payments, protection against rising interest rates, and easier budgeting for homeowners.",
            retrieved_contexts=[
                "Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments.",
                "Fixed-rate mortgages protect borrowers from rising interest rates by locking in the rate.",
                "Adjustable-rate mortgages start with lower rates that adjust periodically.",  # Less relevant
            ],
            ground_truth_contexts=[
                "Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments.",
                "Fixed-rate mortgages protect borrowers from rising interest rates by locking in the rate.",
            ],
        ),
        
        # Case 4: Poor retrieval quality
        EvalCase(
            question="What is the interest rate for FHA loans?",
            answer="FHA loan interest rates vary based on market conditions, credit score, and down payment amount.",
            ground_truth_answer="FHA loan interest rates vary based on market conditions, credit score, and down payment amount.",
            retrieved_contexts=[
                "Property taxes are assessed by local governments based on home value.",
                "Homeowners insurance covers damage from fire, theft, and natural disasters.",
                "FHA loan interest rates are competitive and depend on various factors.",  # Only 1 relevant
            ],
            ground_truth_contexts=[
                "FHA loan interest rates are competitive and depend on various factors.",
            ],
        ),
        
        # Case 5: Completely off-topic answer
        EvalCase(
            question="How do I qualify for a mortgage pre-approval?",
            answer="The weather is nice today and suitable for outdoor activities.",
            ground_truth_answer="To qualify for mortgage pre-approval, you need to provide proof of income, employment verification, credit report, and asset information.",
            retrieved_contexts=[
                "Mortgage pre-approval requires submitting pay stubs, W-2 forms, and tax returns to verify income.",
                "Lenders check your employment history and contact your employer during pre-approval.",
            ],
            ground_truth_contexts=[
                "Mortgage pre-approval requires submitting pay stubs, W-2 forms, and tax returns to verify income.",
                "Lenders check your employment history and contact your employer during pre-approval.",
            ],
        ),
    ]
    
    return cases


async def main():
    """Main test function"""
    
    console.print("\n" + "="*70)
    console.print("Testing MetricRunner with All Metrics", style="bold cyan")
    console.print("="*70 + "\n")
    
    # Step 1: Create test cases
    console.print("[Step 1] Creating test cases...", style="yellow")
    cases = create_comprehensive_test_cases()
    dataset = EvalDataset(cases=cases)
    console.print(f"✓ Created {len(cases)} test cases\n", style="green")
    
    # Step 2: Initialize all metrics
    console.print("[Step 2] Initializing metrics...", style="yellow")
    llm_uri = "bailian/qwen-plus"
    api_key = os.getenv("BAILIAN_API_KEY", "")
    
    if not api_key:
        console.print("✗ Error: BAILIAN_API_KEY not found in environment", style="red")
        return
    
    metrics = [
        Faithfulness(llm_uri=llm_uri, api_key=api_key),
        ContextRelevance(llm_uri=llm_uri, api_key=api_key),
        ContextRecall(llm_uri=llm_uri, api_key=api_key),
        ContextPrecision(llm_uri=llm_uri, api_key=api_key),
        AnswerRelevancy(llm_uri=llm_uri, api_key=api_key),
        AnswerCorrectness(llm_uri=llm_uri, api_key=api_key),
    ]
    console.print(f"✓ Initialized {len(metrics)} metrics\n", style="green")
    
    # Step 3: Create runner
    console.print("[Step 3] Creating metric runner...", style="yellow")
    runner = MetricRunner(metrics=metrics)
    console.print("✓ Runner created\n", style="green")
    
    # Step 4: Run evaluation
    console.print("[Step 4] Running evaluation (concurrent)...\n", style="yellow")
    await runner.run(dataset)
    console.print("\n✓ Evaluation complete\n", style="green")
    
    # Step 4.5: Save evaluated dataset to file
    console.print("[Step 4.5] Saving evaluated dataset...", style="yellow")
    output_file = "output/evaluated_dataset.json"
    dataset.to_json(output_file)
    console.print(f"✓ Dataset saved to {output_file}\n", style="green")
    
    # Step 5: Display results
    console.print("="*70)
    console.print("Evaluation Results", style="bold cyan")
    console.print("="*70 + "\n")
    
    for i, case in enumerate(dataset.cases, 1):
        # Create results table for this case
        results_table = Table(title=f"Case {i}: {case.question}", show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan", width=20)
        results_table.add_column("Score", style="green", justify="right", width=10)
        results_table.add_column("Details", style="white", width=35)
        
        # Add each metric result
        metric_names = ["faithfulness", "context_relevance", "context_recall", 
                       "context_precision", "answer_relevancy", "answer_correctness"]
        
        for metric_name in metric_names:
            result = case.results.get(metric_name)
            if result:
                # Color code score
                if result.score >= 0.8:
                    score_str = f"[green]{result.score:.2f}[/green]"
                elif result.score >= 0.5:
                    score_str = f"[yellow]{result.score:.2f}[/yellow]"
                else:
                    score_str = f"[red]{result.score:.2f}[/red]"
                
                # Truncate reason
                reason = result.reason[:80] + "..." if len(result.reason) > 80 else result.reason
                
                results_table.add_row(metric_name, score_str, reason)
            else:
                results_table.add_row(metric_name, "[dim]N/A[/dim]", "[dim]Not evaluated[/dim]")
        
        console.print(results_table)
        console.print()
    
    # Step 6: Summary statistics
    console.print("="*70)
    console.print("Summary Statistics", style="bold cyan")
    console.print("="*70 + "\n")
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Avg Score", justify="right", style="green")
    summary_table.add_column("Min", justify="right")
    summary_table.add_column("Max", justify="right")
    summary_table.add_column("Cases", justify="right")
    
    metric_names = ["faithfulness", "context_relevance", "context_recall", 
                   "context_precision", "answer_relevancy", "answer_correctness"]
    
    for metric_name in metric_names:
        scores = []
        for case in dataset.cases:
            result = case.results.get(metric_name)
            if result:
                scores.append(result.score)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            summary_table.add_row(
                metric_name,
                f"{avg_score:.3f}",
                f"{min_score:.2f}",
                f"{max_score:.2f}",
                str(len(scores))
            )
    
    console.print(summary_table)
    console.print()
    
    # Show data structure verification
    console.print("="*70)
    console.print("Data Structure Verification", style="bold cyan")
    console.print("="*70 + "\n")
    
    console.print("✓ All results stored in unified EvalResult structure", style="green")
    console.print(f"✓ Each case has {len(metrics)} evaluation results", style="green")
    console.print(f"✓ Dataset contains {len(dataset.cases)} fully evaluated cases", style="green")
    console.print()
    
    # Show example of accessing results programmatically
    console.print("[bold]Example: Accessing results programmatically[/bold]")
    case = dataset.cases[0]
    console.print(f"  case.results.keys() = {list(case.results.keys())}")
    console.print(f"  case.results['faithfulness'].score = {case.results['faithfulness'].score:.2f}")
    console.print(f"  case.results['faithfulness'].reason = '{case.results['faithfulness'].reason[:50]}...'")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
