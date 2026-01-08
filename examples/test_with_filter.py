"""
Test generation with filter enabled

Tests that filter integration works correctly with generators.
"""

import os
import asyncio
from dotenv import load_dotenv
from rich import print as rprint

from zeval.schemas.unit import BaseUnit, UnitType, UnitMetadata
from zeval.synthetic_data.generators.persona import Persona, generate_personas
from zeval.synthetic_data.generators.multi_hop import generate_multi_hop
from zeval.synthetic_data.graphs import KeyphraseOverlapBuilder
from zeval.synthetic_data.filters import GeneralFilter, StrictnessLevel
from pydantic import Field

load_dotenv()


class HomeBuyerPersona(Persona):
    """US Home Buyer persona"""
    credit_score: int = Field(description="Credit score (300-850)")
    dti_ratio: float = Field(description="Debt-to-Income ratio")


async def main():
    """Test multi-hop generation with filter"""
    
    # Check API key
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("Error: BAILIAN_API_KEY environment variable not set")
        return
    
    rprint("\n[bold cyan]Testing Multi-Hop Generation WITH Filter[/bold cyan]\n")
    
    # Create test units
    rprint("[cyan]Step 1: Creating units...[/cyan]")
    units = [
        BaseUnit(
            unit_id="unit_001",
            content="Down payment requirements vary by loan type and credit score. "
                   "FHA loans allow as little as 3.5% down for borrowers with credit scores above 580.",
            unit_type=UnitType.TEXT,
            keyphrases=["down payment", "credit score", "FHA loans"],
            metadata=UnitMetadata(context_path="chapter2/financing")
        ),
        BaseUnit(
            unit_id="unit_002",
            content="Private Mortgage Insurance (PMI) is required when your down payment is less than 20%. "
                   "PMI typically costs 0.5-1% of the loan amount annually.",
            unit_type=UnitType.TEXT,
            keyphrases=["PMI", "down payment", "mortgage insurance"],
            metadata=UnitMetadata(context_path="chapter2/financing")
        ),
        BaseUnit(
            unit_id="unit_003",
            content="Credit scores above 740 qualify for the best mortgage rates. "
                   "Lower credit scores may require larger down payments.",
            unit_type=UnitType.TEXT,
            keyphrases=["credit score", "mortgage rates", "down payment"],
            metadata=UnitMetadata(context_path="chapter1/basics")
        ),
    ]
    rprint(f"[green]✓[/green] Created {len(units)} units\n")
    
    # Build graph
    rprint("[cyan]Step 2: Building graph...[/cyan]")
    builder = KeyphraseOverlapBuilder(threshold=0.1)
    graph = builder.build(units)
    rprint(f"[green]✓[/green] Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")
    
    # Generate personas
    rprint("[cyan]Step 3: Generating personas...[/cyan]")
    personas = await generate_personas(
        llm_uri="bailian/qwen-plus",
        api_key=api_key,
        domain="US home buying",
        num_personas=2,
        persona_model=HomeBuyerPersona
    )
    rprint(f"[green]✓[/green] Generated {len(personas)} personas\n")
    
    # Create filter
    rprint("[cyan]Step 4: Creating filter...[/cyan]")
    filter = GeneralFilter(
        uri="bailian/qwen-plus",
        api_key=api_key,
        concurrency=2,
        strictness=StrictnessLevel.MODERATE  # Allow logical inference for multi-hop
    )
    rprint(f"[green]✓[/green] Filter ready (strictness: {filter.strictness.value})\n")
    
    # Generate with filter
    rprint("[bold cyan]Step 5: Generating cases WITH filter...[/bold cyan]")
    dataset = await generate_multi_hop(
        llm_uri="bailian/qwen-plus",
        api_key=api_key,
        graph=graph,
        personas=personas,
        num_cases=5,
        path_length=2,
        filter=filter  # Enable filtering
    )
    
    rprint(f"\n[bold green]✓ Test completed![/bold green]")
    rprint(f"[green]Final dataset: {len(dataset.cases)} cases (after filtering)[/green]\n")
    
    # Save
    dataset.to_json("tmp/filtered_test_dataset.json")
    rprint(f"[cyan]Saved to tmp/filtered_test_dataset.json[/cyan]\n")


if __name__ == "__main__":
    asyncio.run(main())
