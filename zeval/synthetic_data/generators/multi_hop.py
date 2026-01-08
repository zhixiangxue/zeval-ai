"""Multi-hop question generator

Generate evaluation cases where answers require synthesizing information from multiple related units.
"""

from pydantic import BaseModel, Field
import asyncio
import chak
import networkx as nx
from datetime import datetime
from dataclasses import dataclass

from ...schemas.unit import BaseUnit
from ...schemas.eval import EvalCase, EvalDataset
from ..filters import Filter
from .persona import Persona


class GeneratedQA(BaseModel):
    """LLM output: generated question and answer pair"""
    question: str = Field(description="Generated question requiring multi-hop reasoning")
    answer: str = Field(description="Generated answer synthesizing information from multiple contexts")


@dataclass
class MultiHopScenario:
    """Multi-hop scenario: path through related units"""
    path: list[BaseUnit]  # Units along the path
    theme: str            # Common theme connecting the path
    persona: Persona      # Target user persona


def extract_paths_from_graph(
    graph: nx.Graph,
    path_length: int = 2
) -> list[list[str]]:
    """
    Extract all simple paths of specified length from graph
    
    Args:
        graph: NetworkX graph with units as nodes
        path_length: Desired path length (default: 2)
    
    Returns:
        List of paths, each path is a list of unit_ids
    """
    all_paths = []
    
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                try:
                    # Find all simple paths with length = path_length
                    paths = nx.all_simple_paths(
                        graph, source, target, 
                        cutoff=path_length - 1
                    )
                    for path in paths:
                        if len(path) == path_length:
                            all_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
    
    return all_paths


def extract_common_theme(units: list[BaseUnit]) -> str | None:
    """
    Extract common theme from a path of units (intersection of keyphrases)
    
    Args:
        units: List of units along a path
    
    Returns:
        First common keyphrase, or None if no common theme found
    """
    if not units:
        return None
    
    # Start with first unit's keyphrases
    common = set(units[0].keyphrases or [])
    
    # Intersect with remaining units
    for unit in units[1:]:
        common &= set(unit.keyphrases or [])
    
    # Return first common keyphrase if any
    return list(common)[0] if common else None


def prepare_scenarios(
    graph: nx.Graph,
    personas: list[Persona],
    num_scenarios: int,
    path_length: int = 2
) -> list[MultiHopScenario]:
    """
    Prepare multi-hop scenarios using Ragas-style sampling
    
    Strategy:
    1. Extract all paths of specified length from graph
    2. Extract common theme for each path
    3. Generate all combinations: path × persona
    4. Shuffle to randomize
    5. Sample with constraints:
       - Balanced persona distribution
       - Prefer unique (path, persona) pairs
    
    Args:
        graph: NetworkX graph with units as nodes
        personas: List of user personas
        num_scenarios: Number of scenarios to generate
        path_length: Path length (default: 2)
    
    Returns:
        List of multi-hop scenarios
    """
    import random
    from rich import print as rprint
    
    # Step 1: Extract all paths
    all_paths = extract_paths_from_graph(graph, path_length)
    rprint(f"[dim]  Found {len(all_paths)} paths of length {path_length}[/dim]")
    
    # Step 2: Path → (units, theme)
    valid_paths = []
    for path in all_paths:
        units = [graph.nodes[nid]['unit'] for nid in path]
        theme = extract_common_theme(units)
        
        if theme:  # Only keep paths with common theme
            valid_paths.append((units, theme))
    
    rprint(f"[dim]  {len(valid_paths)} paths have common themes[/dim]")
    
    if not valid_paths:
        rprint(f"[yellow]⚠[/yellow] No valid paths with common themes found!")
        return []
    
    # Step 3: Generate all combinations (path × persona)
    all_combinations = []
    for units, theme in valid_paths:
        for persona in personas:
            all_combinations.append({
                "path": units,
                "theme": theme,
                "persona": persona
            })
    
    # Step 4: Shuffle to randomize
    random.shuffle(all_combinations)
    
    # Step 5: Sample with constraints
    selected = []
    path_persona_set = set()  # Track (path, persona) pairs
    persona_count = {p.name: 0 for p in personas}
    
    # Calculate target count per persona (strict)
    target_per_persona = num_scenarios // len(personas)
    remainder = num_scenarios % len(personas)
    
    for combo in all_combinations:
        if len(selected) >= num_scenarios:
            break
        
        # Create unique key for (path, persona)
        path_ids = tuple(u.unit_id for u in combo["path"])
        persona_name = combo["persona"].name
        key = (path_ids, persona_name)
        
        # Constraint 1: Strict persona balance
        max_for_this_persona = target_per_persona + (1 if remainder > 0 else 0)
        if persona_count[persona_name] >= max_for_this_persona:
            continue
        
        # Constraint 2: Prefer unique (path, persona) pairs
        if key in path_persona_set:
            # Only allow duplicates if running out of options
            if len(selected) < num_scenarios * 0.8:
                continue
        
        # Accept this scenario
        selected.append(MultiHopScenario(
            path=combo["path"],
            theme=combo["theme"],
            persona=combo["persona"]
        ))
        
        path_persona_set.add(key)
        persona_count[persona_name] += 1
        
        # Update remainder counter
        if persona_count[persona_name] == target_per_persona + 1:
            remainder -= 1
    
    return selected


async def generate_multi_hop(
    llm_uri: str,
    api_key: str,
    graph: nx.Graph,
    personas: list[Persona],
    num_cases: int = 10,
    path_length: int = 2,
    domain: str | None = None,
    max_concurrent: int = 5,
    filter: Filter | None = None,  # Optional filter object
) -> EvalDataset:
    """
    Generate multi-hop evaluation dataset from relationship graph
    
    Two-stage approach:
    1. Prepare scenarios: extract paths from graph and match with personas
    2. Generate cases: concurrent generation with rate limiting
    
    Args:
        llm_uri: LLM URI (e.g., "bailian/qwen-plus")
        api_key: API key for the LLM
        graph: NetworkX graph with enriched units as nodes
        personas: List of user personas
        num_cases: Number of cases to generate (default: 10)
        path_length: Length of reasoning paths (default: 2)
        domain: Domain description (optional)
        max_concurrent: Maximum concurrent LLM calls (default: 5)
        filter: Optional Filter object for quality control
    
    Returns:
        EvalDataset containing generated multi-hop evaluation cases
    """
    from rich import print as rprint
    
    # Stage 1: Prepare scenarios
    rprint(f"[cyan]Preparing scenarios for {num_cases} cases (path_length={path_length})...[/cyan]")
    scenarios = prepare_scenarios(graph, personas, num_cases, path_length)
    rprint(f"[green]✓[/green] Prepared {len(scenarios)} valid scenarios")
    
    if not scenarios:
        rprint(f"[red]✗[/red] No valid scenarios found. Check graph connectivity and unit keyphrases.")
        return EvalDataset(cases=[], domain=domain, generated_at=datetime.now().isoformat())
    
    # Stage 2: Generate cases with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def generate_one_case(scenario: MultiHopScenario, index: int) -> EvalCase | None:
        """Generate single multi-hop case with rate limiting"""
        async with semaphore:
            try:
                # Build contexts string with hop markers
                contexts_str = ""
                for i, unit in enumerate(scenario.path, 1):
                    contexts_str += f"<{i}-hop>\n{unit.content}\n\n"
                
                # Build persona info
                persona_info = f"{scenario.persona.name} ({scenario.persona.expertise_level}): {scenario.persona.role_description}"
                
                # Build prompt
                prompt = f"""Generate a multi-hop question-answer pair based on the provided contexts.

Contexts (YOU MUST USE INFORMATION FROM MULTIPLE HOPS):
{contexts_str}

Persona (for context only, DO NOT mention role in question):
{persona_info}

Theme/Focus:
{scenario.theme}

CRITICAL REQUIREMENTS:
1. The question should require information from **MULTIPLE contexts** to answer
2. The answer MUST synthesize information across the {len(scenario.path)} contexts above
3. DO NOT generate a question that can be answered from a single context
4. The question should be relevant to the theme: {scenario.theme}
5. The question should reflect the persona's focus area ({scenario.persona.focus_area}) but **DO NOT explicitly mention the persona's role in the question**
6. The question should be natural and realistic, as if asked by a real person
7. **IMPORTANT**: Write the question directly without "As a ..." prefix - questions should be natural queries

Example of WRONG question: "As a real estate investor, how does X relate to Y?" (too artificial)
Example of RIGHT question: "How does X relate to Y when buying investment properties?" (natural + context-implied)
"""
                
                # Generate QA pair
                result = await conv.asend(prompt, returns=GeneratedQA)
                
                # Build source_units list (all units in path)
                source_units = []
                for unit in scenario.path:
                    source_units.append({
                        "unit_id": unit.unit_id,
                        "unit_type": unit.unit_type,
                        "summary": unit.summary or "",
                        "keyphrases": unit.keyphrases or [],
                        "entities": unit.entities or []
                    })
                
                # Build persona dict
                persona_dict = {
                    "name": scenario.persona.name,
                    "role_description": scenario.persona.role_description,
                    "expertise_level": scenario.persona.expertise_level,
                    "focus_area": scenario.persona.focus_area
                }
                
                # Create evaluation case
                case = EvalCase(
                    question=result.question,
                    ground_truth_answer=result.answer,
                    ground_truth_contexts=[unit.content for unit in scenario.path],
                    answer=None,  # Will be filled during evaluation
                    retrieved_contexts=None,
                    source_units=source_units,
                    persona=persona_dict,
                    generation_params={
                        "generator": "multi_hop",
                        "llm_uri": llm_uri,
                        "theme": scenario.theme,
                        "path_length": len(scenario.path),
                        "generated_at": datetime.now().isoformat()
                    }
                )
                
                return case
                
            except Exception as e:
                rprint(f"[yellow]⚠[/yellow] Case {index+1} failed: {e}")
                return None
    
    # Generate all cases concurrently
    rprint(f"[cyan]Generating {len(scenarios)} cases with max_concurrent={max_concurrent}...[/cyan]")
    tasks = [generate_one_case(scenario, i) for i, scenario in enumerate(scenarios)]
    results = await asyncio.gather(*tasks)
    
    # Filter out failed cases
    cases = [case for case in results if case is not None]
    
    # Report generation results
    success_count = len(cases)
    failed_count = len(scenarios) - len(cases)
    
    if failed_count > 0:
        rprint(f"[yellow]⚠[/yellow] Generation completed: {success_count}/{len(scenarios)} cases succeeded, {failed_count} failed")
    else:
        rprint(f"[green]✓[/green] Successfully generated all {success_count} cases")
    
    # Create dataset
    dataset = EvalDataset(
        cases=cases,
        domain=domain,
        generated_at=datetime.now().isoformat()
    )
        
    # Post-process: apply filter if provided
    if filter is not None:
        rprint("\n[cyan]Applying quality filter...[/cyan]")
        dataset = await filter.filter(dataset)
        
    return dataset
