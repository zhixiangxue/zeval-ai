"""
Test context relevance metric evaluation
"""

import os
import asyncio
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console

from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.evaluation.metrics import ContextRelevance

# Load environment variables
load_dotenv()


async def main():
    """Test context relevance metric"""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    llm_uri = "openai/gpt-4o-mini"
    
    rprint("\n" + "="*60)
    rprint("[bold cyan]Testing Context Relevance Metric[/bold cyan]")
    rprint("="*60)
    
    # Step 1: Create test cases with different relevance levels
    rprint("\n[bold][Step 1][/bold] Creating test cases...")
    
    cases = [
        # Case 1: Fully relevant
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            ground_truth_answer="3.5%",
            ground_truth_contexts=["FHA loans allow 3.5% down."],
            retrieved_contexts=[
                "First-time homebuyers often qualify for lower down payment requirements. "
                "FHA loans allow as little as 3.5% down, while conventional loans typically require 5-10%. "
                "However, a larger down payment can help avoid PMI (Private Mortgage Insurance)."
            ],
        ),
        
        # Case 2: Partially relevant
        EvalCase(
            question="What credit score is needed for FHA loans?",
            ground_truth_answer="At least 580",
            ground_truth_contexts=["FHA loans require at least 580 credit score."],
            retrieved_contexts=[
                "Credit scores play a crucial role in mortgage approval. A score above 740 typically qualifies for the best rates. "
                "Scores between 620-740 may still be approved with higher rates."
                # Missing specific FHA credit score info
            ],
        ),
        
        # Case 3: Not relevant
        EvalCase(
            question="What is the interest rate for FHA loans?",
            ground_truth_answer="Varies by market",
            ground_truth_contexts=["FHA loan rates vary by market conditions."],
            retrieved_contexts=[
                "Closing costs typically range from 2-5% of the home purchase price and include fees for appraisals, inspections, "
                "title insurance, and loan origination."
                # Completely unrelated to FHA interest rates
            ],
        ),
        
        # Case 4: Highly relevant (multiple contexts)
        EvalCase(
            question="What are the benefits of fixed-rate mortgages?",
            ground_truth_answer="Predictable monthly payments and stable interest rate",
            ground_truth_contexts=["Fixed-rate mortgages provide payment stability."],
            retrieved_contexts=[
                "Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments.",
                "This stability helps homeowners budget more effectively and protects against rising interest rates.",
            ],
        ),
        
        # Case 5: Partially relevant with noise
        EvalCase(
            question="How do I qualify for a mortgage pre-approval?",
            ground_truth_answer="Submit financial documents to lender",
            ground_truth_contexts=["Pre-approval requires submitting financial documents."],
            retrieved_contexts=[
                "Pre-approval letters show sellers that you're a serious buyer with financing in place. "
                "Getting pre-approved involves submitting financial documents to a lender.",  # Relevant
                "Adjustable-rate mortgages start with lower rates that adjust periodically."  # Not relevant (noise)
            ],
        ),
    ]
    
    dataset = EvalDataset(cases=cases)
    rprint(f"[green]✓[/green] Created {len(dataset.cases)} test cases")
    
    # Step 2: Initialize context relevance metric
    rprint("\n[bold][Step 2][/bold] Initializing context relevance metric...")
    metric = ContextRelevance(
        llm_uri=llm_uri,
        api_key=api_key,
        name="context_relevance"
    )
    rprint("[green]✓[/green] Metric initialized")
    
    # Step 3: Evaluate cases
    rprint("\n[bold][Step 3][/bold] Evaluating cases...")
    await metric.evaluate_batch(dataset.cases, concurrency=3)
    rprint(f"[green]✓[/green] Evaluated {len(dataset.cases)} cases")
    
    # Step 4: Display results
    console = Console()
    console.print("\n" + "="*60)
    console.print("[bold cyan]Evaluation Results[/bold cyan]")
    console.print("="*60 + "\n")
    
    for i, case in enumerate(dataset.cases, 1):
        result = case.results.get("context_relevance")
        
        if not result:
            console.print(f"[red]Case {i}: No result[/red]\n")
            continue
        
        # Color-code score
        score = result.score
        if score >= 0.75:
            score_color = "green"
            level = "High"
        elif score >= 0.4:
            score_color = "yellow"
            level = "Medium"
        else:
            score_color = "red"
            level = "Low"
        
        # Build content
        context_preview = "\n".join(
            f"  {idx+1}. {ctx[:80]}..." if len(ctx) > 80 else f"  {idx+1}. {ctx}"
            for idx, ctx in enumerate(case.retrieved_contexts[:2])
        )
        if len(case.retrieved_contexts) > 2:
            context_preview += f"\n  [dim]... and {len(case.retrieved_contexts) - 2} more[/dim]"
        
        content = f"""[bold yellow]Question:[/bold yellow]
{case.question}

[bold blue]Retrieved Contexts:[/bold blue]
{context_preview}

[bold {score_color}]Relevance Score: {score:.2f}[/bold {score_color}] ([italic]{level}[/italic])

[bold magenta]Reason:[/bold magenta]
{result.reason}

[bold cyan]Elapsed Time:[/bold cyan] {result.elapsed_time:.2f}s"""
        
        # Add judge details if available
        if result.details:
            judge1 = result.details.get('judge1_rating')
            judge2 = result.details.get('judge2_rating')
            if judge1 is not None or judge2 is not None:
                details_text = f"Judge1: {judge1}/2, Judge2: {judge2}/2"
                content += f"\n\n[bold white]Judge Ratings:[/bold white] {details_text}"
        
        console.print(Panel(
            content,
            title=f"[bold white]Case {i}[/bold white]",
            border_style=score_color,
            padding=(1, 2)
        ))
        console.print()
    
    # Step 5: Summary statistics
    scores = [case.results["context_relevance"].score for case in dataset.cases if "context_relevance" in case.results]
    
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        summary = f"""[bold]Overall Statistics:[/bold]

[yellow]Average Score:[/yellow] {avg_score:.2f}
[green]Max Score:[/green] {max_score:.2f}
[red]Min Score:[/red] {min_score:.2f}
[cyan]Total Cases:[/cyan] {len(scores)}"""
        
        console.print(Panel(
            summary,
            title="[bold white]📊 Summary[/bold white]",
            border_style="blue",
            padding=(1, 2)
        ))


if __name__ == "__main__":
    asyncio.run(main())
