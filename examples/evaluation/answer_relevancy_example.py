"""Test Answer Relevancy metric"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.evaluation.metrics import AnswerRelevancy

# Load environment variables
load_dotenv()

console = Console()


def create_test_cases() -> list[EvalCase]:
    """Create test cases for answer relevancy evaluation"""
    
    cases = [
        # Case 1: Perfect relevancy - direct and focused answer
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            answer="FHA loans require a minimum down payment of 3.5%.",
            ground_truth_answer="",  # Not used in answer relevancy
            retrieved_contexts=[],  # Not used in answer relevancy
            ground_truth_contexts=[],
        ),
        
        # Case 2: Good relevancy - answer with some context
        EvalCase(
            question="What credit score is needed for FHA loans?",
            answer="For FHA loans, you typically need a credit score of at least 580 to qualify for the 3.5% down payment option. However, scores between 500-579 may still qualify with a 10% down payment.",
            ground_truth_answer="",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 3: Moderate relevancy - answer with significant extra info
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            answer="FHA loans are great options for first-time homebuyers! They offer many benefits including flexible credit requirements, lower down payments, and competitive interest rates. Speaking of down payments, FHA loans allow you to put down as little as 3.5%. They're backed by the Federal Housing Administration and have been helping people achieve homeownership for decades.",
            ground_truth_answer="",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 4: Low relevancy - mostly off-topic
        EvalCase(
            question="What is the interest rate for FHA loans?",
            answer="Buying a home is an exciting journey! There are many types of mortgages available, including conventional loans, VA loans, and FHA loans. Each has its own requirements and benefits. It's important to shop around and compare different lenders.",
            ground_truth_answer="",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 5: Zero relevancy - noncommittal/evasive answer
        EvalCase(
            question="How do I qualify for a mortgage pre-approval?",
            answer="Well, it depends on many factors. I'm not sure about the exact requirements. You should probably talk to a lender to get more information.",
            ground_truth_answer="",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
        
        # Case 6: Completely off-topic
        EvalCase(
            question="What are the benefits of fixed-rate mortgages?",
            answer="The weather today is sunny and warm. It's a great day for outdoor activities.",
            ground_truth_answer="",
            retrieved_contexts=[],
            ground_truth_contexts=[],
        ),
    ]
    
    return cases


async def main():
    """Main test function"""
    
    console.print("\n" + "="*60)
    console.print("Testing Answer Relevancy Metric", style="bold cyan")
    console.print("="*60 + "\n")
    
    # Step 1: Create test cases
    console.print("[Step 1] Creating test cases...", style="yellow")
    cases = create_test_cases()
    dataset = EvalDataset(cases=cases)
    console.print(f"✓ Created {len(cases)} test cases\n", style="green")
    
    # Step 2: Initialize metric
    console.print("[Step 2] Initializing answer relevancy metric...", style="yellow")
    llm_uri = "openai/gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key:
        console.print("✗ Error: OPENAI_API_KEY not found in environment", style="red")
        return
    
    metric = AnswerRelevancy(llm_uri=llm_uri, api_key=api_key)
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
        result = case.results.get("answer_relevancy")
        if not result:
            continue
        
        scores.append(result.score)
        
        # Determine relevancy level
        if result.score >= 0.8:
            level = "Excellent"
            level_style = "green"
        elif result.score >= 0.6:
            level = "Good"
            level_style = "bright_green"
        elif result.score >= 0.4:
            level = "Moderate"
            level_style = "yellow"
        elif result.score >= 0.2:
            level = "Poor"
            level_style = "orange1"
        else:
            level = "Irrelevant"
            level_style = "red"
        
        # Create panel for each case
        content = []
        content.append(f"[bold]Question:[/bold]\n{case.question}\n")
        
        # Show answer
        answer_preview = case.answer if len(case.answer) <= 200 else case.answer[:200] + "..."
        content.append(f"[bold]Answer:[/bold]\n{answer_preview}\n")
        
        # Show relevancy score
        content.append(f"[bold]Relevancy Score:[/bold] [{level_style}]{result.score:.2f} ({level})[/{level_style}]\n")
        
        # Show criteria flags
        if result.details:
            content.append(f"[bold]Criteria:[/bold]")
            direct_flag = "✓ Direct" if result.details.get("is_direct") else "✗ Indirect"
            irrelevant_flag = "✗ Has irrelevant info" if result.details.get("has_irrelevant_info") else "✓ Focused"
            noncommittal_flag = "✗ Evasive/vague" if result.details.get("is_noncommittal") else "✓ Substantive"
            
            direct_color = "green" if result.details.get("is_direct") else "red"
            focused_color = "green" if not result.details.get("has_irrelevant_info") else "red"
            substantive_color = "green" if not result.details.get("is_noncommittal") else "red"
            
            content.append(f"  [{direct_color}]{direct_flag}[/{direct_color}]")
            content.append(f"  [{focused_color}]{irrelevant_flag}[/{focused_color}]")
            content.append(f"  [{substantive_color}]{noncommittal_flag}[/{substantive_color}]\n")
        
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
        
        summary_table.add_row("Average Relevancy", f"{sum(scores)/len(scores):.2f}")
        summary_table.add_row("Max Relevancy", f"{max(scores):.2f}")
        summary_table.add_row("Min Relevancy", f"{min(scores):.2f}")
        summary_table.add_row("Total Cases", str(len(scores)))
        
        console.print(summary_table)
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
