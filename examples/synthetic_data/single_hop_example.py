"""
Test single-hop question generation
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic import Field
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console

from zeval.synthetic_data.transforms.extractors import KeyphrasesExtractor, EntitiesExtractor, SummaryExtractor
from zeval.synthetic_data.generators.persona import Persona, generate_personas
from zeval.synthetic_data.generators.single_hop import generate_single_hop
from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.schemas.unit import BaseUnit, UnitType, UnitMetadata

# Load environment variables
load_dotenv()


# Define custom persona
class HomeBuyerPersona(Persona):
    """US Home Buyer persona with financial attributes"""
    credit_score: int = Field(
        description="Credit score (300-850), affects mortgage eligibility and interest rates"
    )
    dti_ratio: float = Field(
        description="Debt-to-Income ratio as percentage (typical max is 43%)"
    )
    down_payment_percent: float = Field(
        description="Down payment as percentage of home price (typically 3-20%)"
    )
    budget_range: str = Field(
        description="Home price budget range (e.g., '$300K-$500K')"
    )


async def main():
    """Test single-hop generation"""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    llm_uri = "openai/gpt-4o-mini"
    
    rprint("\n" + "="*60)
    rprint("[bold cyan]Testing Single-Hop Question Generation[/bold cyan]")
    rprint("="*60)
    
    # Step 1: Create sample units
    rprint("\n[bold][Step 1][/bold] Creating sample units...")
    units = [
        BaseUnit(
            unit_id="unit_001",
            content="First-time homebuyers often qualify for lower down payment requirements. "
                   "FHA loans allow as little as 3.5% down, while conventional loans typically require 5-10%. "
                   "However, a larger down payment can help avoid PMI (Private Mortgage Insurance) and secure better interest rates.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/financing/down_payments")
        ),
        BaseUnit(
            unit_id="unit_002",
            content="Credit scores play a crucial role in mortgage approval. A score above 740 typically qualifies for the best rates, "
                   "while scores between 620-740 may still be approved but with higher interest rates. "
                   "Scores below 620 often require FHA loans or other special programs.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter1/basics/credit_scores")
        ),
        BaseUnit(
            unit_id="unit_003",
            content="The Debt-to-Income (DTI) ratio is a key factor in mortgage qualification. "
                   "Lenders typically prefer a DTI below 43%, though some programs allow up to 50%. "
                   "DTI is calculated by dividing total monthly debt payments by gross monthly income.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/qualification/dti_ratio")
        ),
        BaseUnit(
            unit_id="unit_004",
            content="Closing costs typically range from 2-5% of the home purchase price and include fees for appraisals, inspections, "
                   "title insurance, and loan origination. Buyers should budget for these costs in addition to the down payment. "
                   "Some sellers may offer to cover part of the closing costs as a negotiation strategy.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter3/process/closing_costs")
        ),
        BaseUnit(
            unit_id="unit_005",
            content="Fixed-rate mortgages maintain the same interest rate throughout the loan term, providing predictable monthly payments. "
                   "Adjustable-rate mortgages (ARMs) start with lower rates that adjust periodically based on market conditions. "
                   "ARMs typically have a fixed period (e.g., 5/1 ARM means fixed for 5 years, then adjusts annually).",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/financing/mortgage_types")
        ),
        BaseUnit(
            unit_id="unit_006",
            content="Pre-approval letters show sellers that you're a serious buyer with financing in place. "
                   "Getting pre-approved involves submitting financial documents (pay stubs, tax returns, bank statements) to a lender "
                   "who will verify your income, assets, and creditworthiness. Pre-approval is stronger than pre-qualification.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter1/preparation/pre_approval")
        ),
    ]
    rprint(f"[green]✓[/green] Created {len(units)} units")
    
    # Step 2: Enrich units
    rprint("\n[bold][Step 2][/bold] Enriching units...")
    extractor = (
        SummaryExtractor(model_uri=llm_uri, api_key=api_key)
        | KeyphrasesExtractor(model_uri=llm_uri, api_key=api_key)
        | EntitiesExtractor(model_uri=llm_uri, api_key=api_key)
    )
    enriched_units = await extractor.transform(units)
    rprint(f"[green]✓[/green] Enriched {len(enriched_units)} units")
    
    # Step 3: Generate personas
    rprint("\n[bold][Step 3][/bold] Generating personas...")
    personas = await generate_personas(
        llm_uri=llm_uri,
        api_key=api_key,
        domain="US residential real estate and home buying process",
        num_personas=3,
        persona_model=HomeBuyerPersona
    )
    rprint(f"[green]✓[/green] Generated {len(personas)} personas:")
    for p in personas:
        rprint(f"  • [cyan]{p.name}[/cyan] ([italic]{p.expertise_level}[/italic])")
    
    # Step 4: Generate single-hop questions
    rprint("\n[bold][Step 4][/bold] Generating single-hop questions...")
    dataset = await generate_single_hop(
        llm_uri=llm_uri,
        api_key=api_key,
        units=enriched_units,
        personas=personas,
        num_cases=10,  # Increase to 10 for more diversity
        domain="US residential real estate"
    )
    rprint(f"[green]✓[/green] Generated {len(dataset.cases)} evaluation cases")
    
    # Step 5: Display results with statistics
    console = Console()
    console.print("\n" + "="*60)
    console.print("[bold cyan]Generated Evaluation Cases[/bold cyan]")
    console.print("="*60 + "\n")
    
    # Calculate statistics
    from collections import Counter
    persona_stats = Counter(case.persona['name'] for case in dataset.cases)
    unit_stats = Counter(case.source_units[0]['unit_id'] for case in dataset.cases if case.source_units)
    
    # Display statistics
    persona_dist = "\n".join(f"  • {name}: {count} cases" for name, count in persona_stats.items())
    unit_dist = "\n".join(f"  • {unit_id}: {count} cases" for unit_id, count in unit_stats.items())
    
    stats_content = f"""[bold]Dataset Statistics:[/bold]

[yellow]Total Cases:[/yellow] {len(dataset.cases)}

[cyan]Persona Distribution:[/cyan]
{persona_dist}

[magenta]Source Unit Distribution:[/magenta]
{unit_dist}
"""
    
    console.print(Panel(
        stats_content,
        title="[bold white]📊 Statistics[/bold white]",
        border_style="green",
        padding=(1, 2)
    ))
    console.print()
    
    for i, case in enumerate(dataset.cases, 1):
        # Build case content
        content = f"""[bold yellow]Question:[/bold yellow]
{case.question}

[bold green]Ground Truth Answer:[/bold green]
{case.ground_truth_answer}

[bold blue]AI Answer:[/bold blue]
{case.answer or '[dim](not evaluated yet)[/dim]'}

[bold magenta]Source Unit:[/bold magenta] {case.source_units[0]['unit_id'] if case.source_units else 'N/A'}
[dim]{case.source_units[0].get('summary', 'N/A') if case.source_units else ''}[/dim]

[bold cyan]Persona:[/bold cyan] {case.persona['name']} ([italic]{case.persona['expertise_level']}[/italic])"""
        
        console.print(Panel(
            content,
            title=f"[bold white]Case {i}[/bold white]",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print()
    
    # Step 6: Export dataset
    rprint("\n[bold][Step 6][/bold] Exporting dataset...")
    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
        
    dataset.to_json(f"{output_dir}/eval_dataset.json")
    rprint(f"[green]✓[/green] Exported to [cyan]{output_dir}/eval_dataset.json[/cyan]")
        
    dataset.to_jsonl(f"{output_dir}/eval_dataset.jsonl")
    rprint(f"[green]✓[/green] Exported to [cyan]{output_dir}/eval_dataset.jsonl[/cyan]")
        
    dataset.to_csv(f"{output_dir}/eval_dataset.csv")
    rprint(f"[green]✓[/green] Exported to [cyan]{output_dir}/eval_dataset.csv[/cyan]")
        
    rprint("\n" + "="*60)
    rprint("[bold green]✓ ALL TESTS COMPLETED![/bold green]")
    rprint("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
