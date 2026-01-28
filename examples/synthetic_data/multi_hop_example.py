"""
Test multi-hop question generation
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
from zeval.synthetic_data.generators.multi_hop import generate_multi_hop
from zeval.synthetic_data.graphs import KeyphraseOverlapBuilder
from zeval.synthetic_data.filters import GeneralFilter
from zeval.schemas.eval import EvalCase, EvalDataset
from zeval.schemas.unit import BaseUnit, UnitType, UnitMetadata

# Load environment variables
load_dotenv()


# Define custom persona (same as single-hop)
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
    """Test multi-hop generation"""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    llm_uri = "openai/gpt-4o-mini"
    
    rprint("\n" + "="*60)
    rprint("[bold cyan]Testing Multi-Hop Question Generation[/bold cyan]")
    rprint("="*60)
    
    # Step 1: Create sample units (designed for keyphrase overlap)
    rprint("\n[bold][Step 1][/bold] Creating sample units...")
    units = [
        BaseUnit(
            unit_id="unit_001",
            content="Down payment requirements vary by loan type and credit score. "
                   "FHA loans allow as little as 3.5% down for borrowers with credit scores above 580. "
                   "Conventional loans typically require 5-20% down, with the exact amount influenced by your credit score. "
                   "A higher down payment reduces your loan amount and may help you avoid Private Mortgage Insurance (PMI). "
                   "Lenders also consider your debt-to-income ratio when determining down payment requirements.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/financing/down_payments")
        ),
        BaseUnit(
            unit_id="unit_002",
            content="Credit scores are crucial for mortgage approval and determine your interest rates. "
                   "A credit score above 740 qualifies for the best mortgage rates, while scores between 620-740 face higher rates. "
                   "Your credit score directly affects the loan programs available to you and the down payment percentage required. "
                   "Lower credit scores may require larger down payments to offset lending risk. "
                   "Improving your credit score before applying can significantly reduce your total borrowing costs.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter1/basics/credit_scores")
        ),
        BaseUnit(
            unit_id="unit_003",
            content="The Debt-to-Income (DTI) ratio measures your monthly debt payments against your gross monthly income. "
                   "Lenders typically prefer a DTI below 43%, though some loan programs allow up to 50%. "
                   "Your DTI ratio works together with your credit score to determine loan eligibility. "
                   "A lower DTI ratio may allow you to qualify for a larger loan amount or better interest rates. "
                   "Paying down existing debts can improve your DTI ratio and strengthen your mortgage application.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/qualification/dti_ratio")
        ),
        BaseUnit(
            unit_id="unit_004",
            content="Closing costs typically range from 2-5% of the home purchase price and include various fees. "
                   "Common closing costs include appraisal fees, inspection fees, title insurance, and loan origination fees. "
                   "Your down payment and closing costs together determine your total upfront cash needed. "
                   "Some lenders offer no-closing-cost mortgages, but these come with higher interest rates. "
                   "Sellers may negotiate to cover part of closing costs, reducing your cash requirement at closing.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter3/process/closing_costs")
        ),
        BaseUnit(
            unit_id="unit_005",
            content="Private Mortgage Insurance (PMI) is required when your down payment is less than 20% of the home price. "
                   "PMI protects the lender if you default, with monthly costs typically 0.5-1% of the loan amount annually. "
                   "You can remove PMI once you reach 20% equity through payments or home appreciation. "
                   "A larger down payment helps you avoid PMI and reduces your monthly mortgage payment. "
                   "FHA loans have their own mortgage insurance (MIP) that may be required for the life of the loan.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/financing/pmi")
        ),
        BaseUnit(
            unit_id="unit_006",
            content="Interest rates on mortgages are influenced by your credit score, down payment amount, and loan type. "
                   "Fixed-rate mortgages maintain the same interest rate throughout the entire loan term, typically 15 or 30 years. "
                   "Borrowers with higher credit scores and larger down payments qualify for lower interest rates. "
                   "Your interest rate directly impacts your monthly payment amount and total interest paid over the loan's life. "
                   "Even a 0.25% difference in interest rate can mean thousands of dollars over 30 years.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/financing/interest_rates")
        ),
        BaseUnit(
            unit_id="unit_007",
            content="Pre-approval is a critical step where lenders evaluate your creditworthiness before you house hunt. "
                   "The pre-approval process examines your credit score, income, assets, and debt-to-income ratio. "
                   "A pre-approval letter shows sellers you're a serious buyer with verified financing capability. "
                   "Getting pre-approved helps you understand your budget and strengthens your offer in competitive markets. "
                   "Pre-approval typically remains valid for 60-90 days, after which it may need renewal.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter1/preparation/pre_approval")
        ),
        BaseUnit(
            unit_id="unit_008",
            content="Loan-to-Value (LTV) ratio compares your loan amount to the home's appraised value or purchase price. "
                   "A lower LTV ratio (achieved through a larger down payment) results in better loan terms and interest rates. "
                   "LTV above 80% typically triggers PMI requirements, increasing your monthly costs. "
                   "Lenders use LTV ratio along with credit score and DTI ratio to assess lending risk. "
                   "Maintaining an LTV below 80% is a key goal for avoiding extra insurance costs.",
            unit_type=UnitType.TEXT,
            metadata=UnitMetadata(context_path="chapter2/qualification/ltv_ratio")
        ),
    ]
    rprint(f"[green]✓[/green] Created {len(units)} units")
    
    # Step 2: Enrich units
    rprint("\n[bold][Step 2][/bold] Enriching units...")
    extractor = (
        SummaryExtractor(model_uri=llm_uri, api_key=api_key)
        | KeyphrasesExtractor(model_uri=llm_uri, api_key=api_key, max_num=5)
        | EntitiesExtractor(model_uri=llm_uri, api_key=api_key, max_num=5)
    )
    enriched_units = await extractor.transform(units)
    rprint(f"[green]✓[/green] Enriched {len(enriched_units)} units")
    
    # Display extracted keyphrases (for debugging)
    rprint("\n[dim]Extracted keyphrases:[/dim]")
    for unit in enriched_units:
        if unit.keyphrases:
            rprint(f"[dim]  {unit.unit_id}: {unit.keyphrases}[/dim]")
    
    # Step 3: Build relationship graph
    rprint("\n[bold][Step 3][/bold] Building relationship graph...")
    graph_builder = KeyphraseOverlapBuilder(threshold=0.1)  # Lower threshold for more connections
    graph = graph_builder.build(enriched_units)
    rprint(f"[green]✓[/green] Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Display graph edges (for debugging)
    if graph.number_of_edges() > 0:
        rprint("\n[dim]Graph edges:[/dim]")
        for u, v, data in graph.edges(data=True):
            unit_u = graph.nodes[u]['unit']
            unit_v = graph.nodes[v]['unit']
            weight = data.get('weight', 0)
            rprint(f"[dim]  {unit_u.unit_id} <-> {unit_v.unit_id} (weight: {weight:.2f})[/dim]")
    else:
        rprint("[yellow]⚠[/yellow] No edges in graph! Consider lowering threshold or enriching content.")
    
    # Step 4: Generate personas
    rprint("\n[bold][Step 4][/bold] Generating personas...")
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
    
    # Step 5: Generate multi-hop questions
    rprint("\n[bold][Step 5][/bold] Generating multi-hop questions...")
    
    # Optional: Enable strict filtering (uncomment to test)
    # filter = GeneralFilter(
    #     uri="openai/gpt-4o",  # Use larger model for judgment
    #     api_key=api_key,
    #     concurrency=3
    # )
    # rprint("[yellow]⚠ Filter enabled: will apply strict quality control[/yellow]")
    
    dataset = await generate_multi_hop(
        llm_uri=llm_uri,
        api_key=api_key,
        graph=graph,
        personas=personas,
        num_cases=12,  # Increase to get more diverse paths
        path_length=2,  # 2-hop reasoning
        domain="US residential real estate",
        # filter=filter  # Uncomment to enable filtering
    )
    rprint(f"[green]✓[/green] Generated {len(dataset.cases)} evaluation cases")
    
    # Step 6: Display results with statistics
    console = Console()
    console.print("\n" + "="*60)
    console.print("[bold cyan]Generated Multi-Hop Evaluation Cases[/bold cyan]")
    console.print("="*60 + "\n")
    
    # Calculate statistics
    from collections import Counter
    persona_stats = Counter(case.persona['name'] for case in dataset.cases)
    
    # Count unique paths (using first and last unit)
    path_stats = Counter()
    for case in dataset.cases:
        if len(case.source_units) >= 2:
            path_key = f"{case.source_units[0]['unit_id']} -> {case.source_units[-1]['unit_id']}"
            path_stats[path_key] += 1
    
    # Display statistics
    persona_dist = "\n".join(f"  • {name}: {count} cases" for name, count in persona_stats.items())
    path_dist = "\n".join(f"  • {path}: {count} cases" for path, count in path_stats.items())
    
    stats_content = f"""[bold]Dataset Statistics:[/bold]

[yellow]Total Cases:[/yellow] {len(dataset.cases)}
[yellow]Path Length:[/yellow] 2-hop

[cyan]Persona Distribution:[/cyan]
{persona_dist}

[magenta]Reasoning Paths:[/magenta]
{path_dist}
"""
    
    console.print(Panel(
        stats_content,
        title="[bold white]📊 Statistics[/bold white]",
        border_style="green",
        padding=(1, 2)
    ))
    console.print()
    
    # Display individual cases
    for i, case in enumerate(dataset.cases, 1):
        # Build path visualization
        path_viz = " -> ".join(unit['unit_id'] for unit in case.source_units)
        
        # Build contexts display (show all hops)
        contexts_display = ""
        for j, context in enumerate(case.ground_truth_contexts, 1):
            # Truncate long contexts
            context_preview = context[:150] + "..." if len(context) > 150 else context
            contexts_display += f"\n[dim]<{j}-hop>[/dim] {context_preview}\n"
        
        # Build case content
        content = f"""[bold yellow]Question:[/bold yellow]
{case.question}

[bold green]Ground Truth Answer:[/bold green]
{case.ground_truth_answer}

[bold blue]AI Answer:[/bold blue]
{case.answer or '[dim](not evaluated yet)[/dim]'}

[bold magenta]Reasoning Path:[/bold magenta] {path_viz}

[bold]Ground Truth Contexts:[/bold]{contexts_display}

[bold cyan]Persona:[/bold cyan] {case.persona['name']} ([italic]{case.persona['expertise_level']}[/italic])

[bold]Theme:[/bold] {case.generation_params.get('theme', 'N/A')}"""
        
        console.print(Panel(
            content,
            title=f"[bold white]Case {i} (Multi-Hop)[/bold white]",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print()
    
    # Step 7: Export dataset
    rprint("\n[bold][Step 7][/bold] Exporting dataset...")
    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
        
    dataset.to_json(f"{output_dir}/multi_hop_dataset.json")
    rprint(f"[green]✓[/green] Exported to [cyan]{output_dir}/multi_hop_dataset.json[/cyan]")
        
    dataset.to_jsonl(f"{output_dir}/multi_hop_dataset.jsonl")
    rprint(f"[green]✓[/green] Exported to [cyan]{output_dir}/multi_hop_dataset.jsonl[/cyan]")
        
    dataset.to_csv(f"{output_dir}/multi_hop_dataset.csv")
    rprint(f"[green]✓[/green] Exported to [cyan]{output_dir}/multi_hop_dataset.csv[/cyan]")
        
    rprint("\n" + "="*60)
    rprint("[bold green]✓ ALL TESTS COMPLETED![/bold green]")
    rprint("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
