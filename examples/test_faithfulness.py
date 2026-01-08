"""
Test faithfulness metric evaluation
"""

import os
import asyncio
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console

from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.evaluation.metrics import Faithfulness

# Load environment variables
load_dotenv()


async def main():
    """Test faithfulness metric"""
    
    # Check for API key
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("Error: BAILIAN_API_KEY environment variable not set")
        return
    
    llm_uri = "bailian/qwen-plus"
    
    rprint("\n" + "="*60)
    rprint("[bold cyan]Testing Faithfulness Metric[/bold cyan]")
    rprint("="*60)
    
    # Step 1: Create test cases with different faithfulness levels
    rprint("\n[bold][Step 1][/bold] Creating test cases...")
    
    cases = [
        # Case 1: Fully faithful (all info from context)
        EvalCase(
            question="What is the minimum down payment for FHA loans?",
            ground_truth_answer="FHA loans allow as little as 3.5% down payment.",
            ground_truth_contexts=[
                "First-time homebuyers often qualify for lower down payment requirements. "
                "FHA loans allow as little as 3.5% down, while conventional loans typically require 5-10%. "
                "However, a larger down payment can help avoid PMI (Private Mortgage Insurance) and secure better interest rates."
            ],
            answer="FHA loans allow as little as 3.5% down payment.",
            retrieved_contexts=[
                "First-time homebuyers often qualify for lower down payment requirements. "
                "FHA loans allow as little as 3.5% down, while conventional loans typically require 5-10%. "
                "However, a larger down payment can help avoid PMI (Private Mortgage Insurance) and secure better interest rates."
            ],
        ),
        
        # Case 2: Partially faithful (some hallucination)
        EvalCase(
            question="What credit score is needed for a mortgage?",
            ground_truth_answer="A score above 740 qualifies for the best rates, scores between 620-740 may get approved with higher rates, and scores below 620 often require FHA loans.",
            ground_truth_contexts=[
                "Credit scores play a crucial role in mortgage approval. A score above 740 typically qualifies for the best rates, "
                "while scores between 620-740 may still be approved but with higher interest rates. "
                "Scores below 620 often require FHA loans or other special programs."
            ],
            answer="A credit score above 740 qualifies for the best rates. "
                   "Scores between 620-740 get approved with higher rates. "
                   "Scores below 600 are automatically rejected.",  # HALLUCINATION: "below 600 automatically rejected" not in context
            retrieved_contexts=[
                "Credit scores play a crucial role in mortgage approval. A score above 740 typically qualifies for the best rates, "
                "while scores between 620-740 may still be approved but with higher interest rates. "
                "Scores below 620 often require FHA loans or other special programs."
            ],
        ),
        
        # Case 3: Low faithfulness (significant hallucination)
        EvalCase(
            question="What are closing costs?",
            ground_truth_answer="Closing costs typically range from 2-5% of the home purchase price and include fees for appraisals, inspections, title insurance, and loan origination.",
            ground_truth_contexts=[
                "Closing costs typically range from 2-5% of the home purchase price and include fees for appraisals, inspections, "
                "title insurance, and loan origination. Buyers should budget for these costs in addition to the down payment. "
                "Some sellers may offer to cover part of the closing costs as a negotiation strategy."
            ],
            answer="Closing costs are fees paid at the end of a home purchase. "
                   "They typically include appraisal fees, inspection fees, and title insurance. "
                   "The federal government mandates that closing costs cannot exceed 3% of the home price. "  # HALLUCINATION
                   "Buyers can negotiate with sellers to split closing costs 50-50.",  # HALLUCINATION
            retrieved_contexts=[
                "Closing costs typically range from 2-5% of the home purchase price and include fees for appraisals, inspections, "
                "title insurance, and loan origination. Buyers should budget for these costs in addition to the down payment. "
                "Some sellers may offer to cover part of the closing costs as a negotiation strategy."
            ],
        ),
        
        # Case 4: Completely faithful (complex multi-fact)
        EvalCase(
            question="What is the difference between fixed-rate and adjustable-rate mortgages?",
            ground_truth_answer="Fixed-rate mortgages maintain the same interest rate throughout the loan term, while ARMs start with lower rates that adjust periodically.",
            ground_truth_contexts=[
                "Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments. "
                "Adjustable-rate mortgages (ARMs) start with lower rates that adjust periodically based on market conditions. "
                "ARMs typically have a fixed period (e.g., 5/1 ARM means fixed for 5 years, then adjusts annually)."
            ],
            answer="Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments. "
                   "Adjustable-rate mortgages (ARMs) start with lower rates that adjust periodically based on market conditions. "
                   "For example, a 5/1 ARM means the rate is fixed for 5 years, then adjusts annually.",
            retrieved_contexts=[
                "Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments. "
                "Adjustable-rate mortgages (ARMs) start with lower rates that adjust periodically based on market conditions. "
                "ARMs typically have a fixed period (e.g., 5/1 ARM means fixed for 5 years, then adjusts annually)."
            ],
        ),
        
        # Case 5: No context available
        EvalCase(
            question="What is the best time to buy a house?",
            ground_truth_answer="The best time depends on various factors.",
            ground_truth_contexts=["Market conditions and personal finances are key considerations."],
            answer="The best time to buy a house depends on market conditions and personal finances.",
            retrieved_contexts=[],  # Empty context
        ),
    ]
    
    dataset = EvalDataset(cases=cases)
    rprint(f"[green]✓[/green] Created {len(dataset.cases)} test cases")
    
    # Step 2: Initialize faithfulness metric
    rprint("\n[bold][Step 2][/bold] Initializing faithfulness metric...")
    metric = Faithfulness(
        llm_uri=llm_uri,
        api_key=api_key,
        name="faithfulness"
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
        result = case.results.get("faithfulness")
        
        if not result:
            console.print(f"[red]Case {i}: No result[/red]\n")
            continue
        
        # Color-code score
        score = result.score
        if score >= 0.8:
            score_color = "green"
            level = "High"
        elif score >= 0.5:
            score_color = "yellow"
            level = "Medium"
        else:
            score_color = "red"
            level = "Low"
        
        # Build content
        context_preview = case.retrieved_contexts[0][:100] + "..." if case.retrieved_contexts else "[dim](no context)[/dim]"
        
        content = f"""[bold yellow]Question:[/bold yellow]
{case.question}

[bold green]Answer:[/bold green]
{case.answer}

[bold blue]Context Preview:[/bold blue]
{context_preview}

[bold {score_color}]Faithfulness Score: {score:.2f}[/bold {score_color}] ([italic]{level}[/italic])

[bold magenta]Reason:[/bold magenta]
{result.reason}

[bold cyan]Elapsed Time:[/bold cyan] {result.elapsed_time:.2f}s"""
        
        # Add details if available
        if result.details and 'statements' in result.details:
            statements = result.details['statements']
            if statements:
                statements_text = "\n".join(
                    f"  {'✓' if s['supported'] else '✗'} {s['text']}"
                    for s in statements[:5]  # Show first 5
                )
                if len(statements) > 5:
                    statements_text += f"\n  [dim]... and {len(statements) - 5} more[/dim]"
                content += f"\n\n[bold white]Statements:[/bold white]\n{statements_text}"
        
        console.print(Panel(
            content,
            title=f"[bold white]Case {i}[/bold white]",
            border_style=score_color,
            padding=(1, 2)
        ))
        console.print()
    
    # Step 5: Summary statistics
    scores = [case.results["faithfulness"].score for case in dataset.cases if "faithfulness" in case.results]
    
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
