"""Test Answer Correctness metric"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.evaluation.metrics import AnswerCorrectness

# Load environment variables
load_dotenv()

console = Console()


def create_test_cases() -> list[EvalCase]:
    """Create test cases for answer correctness evaluation"""
    
    cases = [
        # Case 1: Perfect correctness - complete and accurate
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%.",
            answer="FHA loans require a minimum down payment of 3.5%.",
            retrieved_contexts=[],  # Not used in answer correctness
            ground_truth_contexts=[],
        ),
        
        # Case 2: High correctness - semantically equivalent
        EvalCase(
            question="What credit score is needed for FHA loans?",
            ground_truth_answer="You need a credit score of at least 580 to qualify for FHA loans with 3.5% down payment.",
            answer="FHA loans require a minimum credit score of 580 for the 3.5% down payment option.",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 3: Partial correctness - some correct, some missing
        EvalCase(
            question="What are the requirements for FHA loans?",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%, a credit score of at least 580, and mortgage insurance premiums.",
            answer="FHA loans require a minimum down payment of 3.5% and a credit score of at least 580.",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 4: Low correctness - contains incorrect information
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            ground_truth_answer="FHA loans require a minimum down payment of 3.5%.",
            answer="FHA loans require a minimum down payment of 5% or 10% depending on your credit score.",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 5: Very low correctness - mostly incorrect
        EvalCase(
            question="What powers the sun?",
            ground_truth_answer="The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium, releasing tremendous energy.",
            answer="The sun is powered by nuclear fission, similar to nuclear power plants on Earth.",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 6: Partial - correct but incomplete
        EvalCase(
            question="What are the benefits of fixed-rate mortgages?",
            ground_truth_answer="Fixed-rate mortgages offer predictable monthly payments, protection against rising interest rates, and easier budgeting for homeowners.",
            answer="Fixed-rate mortgages offer predictable monthly payments.",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
    ]
    
    return cases


async def main():
    """Main test function"""
    
    console.print("\n" + "="*60)
    console.print("Testing Answer Correctness Metric", style="bold cyan")
    console.print("="*60 + "\n")
    
    # Step 1: Create test cases
    console.print("[Step 1] Creating test cases...", style="yellow")
    cases = create_test_cases()
    dataset = EvalDataset(cases=cases)
    console.print(f"✓ Created {len(cases)} test cases\n", style="green")
    
    # Step 2: Initialize metric
    console.print("[Step 2] Initializing answer correctness metric...", style="yellow")
    llm_uri = "openai/gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key:
        console.print("✗ Error: OPENAI_API_KEY not found in environment", style="red")
        return
    
    metric = AnswerCorrectness(llm_uri=llm_uri, api_key=api_key)
    console.print("✓ Metric initialized\n", style="green")
    
    # Step 3: Evaluate cases
    console.print("[Step 3] Evaluating cases...", style="yellow")
    await metric.evaluate_batch(dataset.cases, concurrency=2)  # Lower concurrency for complex operation
    console.print(f"✓ Evaluated {len(cases)} cases\n", style="green")
    
    # Step 4: Display results
    console.print("="*60)
    console.print("Evaluation Results", style="bold cyan")
    console.print("="*60 + "\n")
    
    scores = []
    
    for i, case in enumerate(dataset.cases, 1):
        result = case.results.get("answer_correctness")
        if not result:
            continue
        
        scores.append(result.score)
        
        # Determine correctness level
        if result.score >= 0.9:
            level = "Excellent"
            level_style = "green"
        elif result.score >= 0.7:
            level = "Good"
            level_style = "bright_green"
        elif result.score >= 0.5:
            level = "Moderate"
            level_style = "yellow"
        elif result.score >= 0.3:
            level = "Poor"
            level_style = "orange1"
        else:
            level = "Incorrect"
            level_style = "red"
        
        # Create panel for each case
        content = []
        content.append(f"[bold]Question:[/bold]\n{case.question}\n")
        
        # Show answer vs ground truth
        answer_preview = case.answer if len(case.answer) <= 150 else case.answer[:150] + "..."
        gt_preview = case.ground_truth_answer if len(case.ground_truth_answer) <= 150 else case.ground_truth_answer[:150] + "..."
        
        content.append(f"[bold]Answer:[/bold]\n{answer_preview}\n")
        content.append(f"[bold]Ground Truth:[/bold]\n{gt_preview}\n")
        
        # Show F1 score
        content.append(f"[bold]Correctness Score:[/bold] [{level_style}]{result.score:.2f} ({level})[/{level_style}]\n")
        
        # Show metrics breakdown
        if result.details:
            precision = result.details.get("precision", 0)
            recall = result.details.get("recall", 0)
            tp = result.details.get("tp_count", 0)
            fp = result.details.get("fp_count", 0)
            fn = result.details.get("fn_count", 0)
            
            content.append(f"[bold]Metrics:[/bold]")
            content.append(f"  Precision: {precision:.2f} (accuracy of provided info)")
            content.append(f"  Recall: {recall:.2f} (completeness of answer)")
            content.append(f"  TP={tp} (correct), FP={fp} (incorrect), FN={fn} (missing)\n")
        
        # Show reason
        content.append(f"[bold]Analysis:[/bold]\n{result.reason}\n")
        
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
        
        summary_table.add_row("Average F1 Score", f"{sum(scores)/len(scores):.2f}")
        summary_table.add_row("Max Score", f"{max(scores):.2f}")
        summary_table.add_row("Min Score", f"{min(scores):.2f}")
        summary_table.add_row("Total Cases", str(len(scores)))
        
        console.print(summary_table)
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
