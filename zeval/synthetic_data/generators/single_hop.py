"""Single-hop question generator

Generate evaluation cases where answers come from a single unit's content.
"""

from pydantic import BaseModel, Field
import asyncio
import chak
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ...schemas.unit import BaseUnit
from ...schemas.eval import EvalCase, EvalDataset
from .persona import Persona


class GeneratedQA(BaseModel):
    """LLM output: generated question and answer pair"""
    question: str = Field(description="Generated question")
    answer: str = Field(description="Generated answer")


def validate_qa_context_match(question: str, answer: str, context: str) -> tuple[bool, str]:
    """
    Validate if the answer can be derived from the context
    
    Simple heuristic: check if key answer words appear in context
    Returns: (is_valid, reason)
    """
    # Extract key terms from answer (words longer than 4 chars)
    answer_words = set(
        word.lower().strip('.,!?;:') 
        for word in answer.split() 
        if len(word) > 4 and word.isalpha()
    )
    
    context_lower = context.lower()
    
    # Count how many answer key terms appear in context
    matched_words = [word for word in answer_words if word in context_lower]
    match_ratio = len(matched_words) / len(answer_words) if answer_words else 0
    
    # Require at least 30% of key terms to appear in context
    if match_ratio < 0.3:
        return False, f"Only {match_ratio:.0%} of answer key terms found in context"
    
    return True, f"{match_ratio:.0%} match"


@dataclass
class Scenario:
    """Single-hop scenario: specific combination for generation"""
    unit: BaseUnit
    persona: Persona
    theme: str  # Focus theme (from keyphrases or entities)


def prepare_scenarios(
    units: list[BaseUnit],
    personas: list[Persona],
    num_scenarios: int
) -> list[Scenario]:
    """
    Prepare scenarios using Ragas-style exhaustive combination + shuffle
    
    Strategy (inspired by Ragas):
    1. Generate all possible combinations: unit × theme × persona
    2. Shuffle to randomize
    3. Sample with strict constraints:
       - Each unit generates at most 2 questions (avoid DTI-like repetition)
       - Balanced persona distribution (strict round-robin)
       - Prefer unique (unit, persona) pairs
    """
    import random
    
    # Step 1: Generate all combinations
    all_combinations = []
    
    for unit in units:
        # Extract themes from unit
        themes = []
        if unit.keyphrases:
            themes.extend(unit.keyphrases)
        if unit.entities:
            themes.extend(unit.entities)
        
        # Fallback: use summary as theme
        if not themes:
            if unit.summary:
                themes = [unit.summary[:50]]  # First part of summary
            else:
                themes = ["general topic"]  # Ultimate fallback
        
        # Create all unit × theme × persona combinations
        for theme in themes:
            for persona in personas:
                all_combinations.append({
                    "unit": unit,
                    "persona": persona,
                    "theme": theme
                })
    
    # Step 2: Shuffle to randomize
    random.shuffle(all_combinations)
    
    # Step 3: Sample with strict constraints
    selected = []
    unit_persona_set = set()  # Track (unit, persona) pairs
    unit_count = {}  # Track how many times each unit is used
    persona_count = {p.name: 0 for p in personas}  # Track persona usage
    
    # Calculate target count per persona (strict)
    target_per_persona = num_scenarios // len(personas)
    remainder = num_scenarios % len(personas)
    
    # Maximum questions per unit (avoid DTI-like repetition)
    MAX_QUESTIONS_PER_UNIT = 2
    
    for combo in all_combinations:
        if len(selected) >= num_scenarios:
            break
        
        unit = combo["unit"]
        persona = combo["persona"]
        key = (unit.unit_id, persona.name)
        
        # Constraint 1: Each unit can only generate MAX_QUESTIONS_PER_UNIT questions
        unit_usage = unit_count.get(unit.unit_id, 0)
        if unit_usage >= MAX_QUESTIONS_PER_UNIT:
            continue
        
        # Constraint 2: Strict persona balance
        # Allow +1 for remainder distribution
        max_for_this_persona = target_per_persona + (1 if remainder > 0 else 0)
        if persona_count[persona.name] >= max_for_this_persona:
            continue
        
        # Constraint 3: Prefer unique (unit, persona) pairs
        if key in unit_persona_set:
            # Only allow duplicates if we're running out of options
            if len(selected) < num_scenarios * 0.8:  # Still in early phase
                continue
        
        # Accept this scenario
        selected.append(Scenario(
            unit=unit,
            persona=persona,
            theme=combo["theme"]
        ))
        
        unit_persona_set.add(key)
        unit_count[unit.unit_id] = unit_usage + 1
        persona_count[persona.name] += 1
        
        # Update remainder counter
        if persona_count[persona.name] == target_per_persona + 1:
            remainder -= 1
    
    return selected


async def generate_single_hop(
    llm_uri: str,
    api_key: str,
    units: list[BaseUnit],
    personas: list[Persona],
    num_cases: int = 10,
    domain: str | None = None,
    max_concurrent: int = 5,  # Concurrency limit
) -> EvalDataset:
    """
    Generate single-hop evaluation dataset with scenario-based sampling
    
    Two-stage approach:
    1. Prepare scenarios: match personas with units based on themes
    2. Generate cases: concurrent generation with rate limiting
    
    Args:
        llm_uri: LLM URI (e.g., "bailian/qwen-plus")
        api_key: API key for the LLM
        units: List of enriched units
        personas: List of user personas
        num_cases: Number of cases to generate (default: 10)
        domain: Domain description (optional)
        max_concurrent: Maximum concurrent LLM calls (default: 5)
    
    Returns:
        EvalDataset containing generated evaluation cases
    """
    from rich import print as rprint
    
    # Stage 1: Prepare scenarios
    rprint(f"[cyan]Preparing scenarios for {num_cases} cases...[/cyan]")
    scenarios = prepare_scenarios(units, personas, num_cases)
    rprint(f"[green]✓[/green] Prepared {len(scenarios)} valid scenarios")
    
    # Stage 2: Generate cases with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def generate_one_case(scenario: Scenario, index: int) -> EvalCase | None:
        """Generate single case with rate limiting"""
        async with semaphore:
            try:
                # Build prompt with theme
                persona_info = f"{scenario.persona.name} ({scenario.persona.expertise_level}): {scenario.persona.role_description}"
                
                prompt = f"""Generate a question-answer pair based STRICTLY on the provided context.

Context (THIS IS YOUR ONLY SOURCE OF TRUTH):
{scenario.unit.content}

Persona:
{persona_info}

Theme/Focus (use as inspiration, but ONLY answer from context):
{scenario.theme}

CRITICAL REQUIREMENTS:
1. The question should be relevant to the theme: {scenario.theme}
2. The question should match the persona's focus area: {scenario.persona.focus_area}
3. **The answer MUST ONLY use information from the context above**
4. **If the context doesn't contain information to answer the question, DO NOT generate this pair**
5. **DO NOT add any information not explicitly stated in the context**
6. The question should be natural and realistic

Example of WRONG answer: Adding information beyond the context
Example of RIGHT answer: Only using facts directly from the context
"""
                
                # Generate QA pair
                result = await conv.asend(prompt, returns=GeneratedQA)
                
                # Validate answer-context match
                is_valid, reason = validate_qa_context_match(
                    result.question, 
                    result.answer, 
                    scenario.unit.content
                )
                
                if not is_valid:
                    rprint(f"[yellow]⚠[/yellow] Case {index+1} validation failed: {reason}")
                    return None
                
                # Build source_unit dict
                source_unit = {
                    "unit_id": scenario.unit.unit_id,
                    "unit_type": scenario.unit.unit_type,
                    "summary": scenario.unit.summary,
                    "keyphrases": scenario.unit.keyphrases or [],
                    "entities": scenario.unit.entities or [],
                }
                if hasattr(scenario.unit, 'metadata') and scenario.unit.metadata:
                    source_unit["context_path"] = scenario.unit.metadata.context_path
                
                # Build persona dict
                persona_dict = scenario.persona.model_dump()
                
                # Create EvalCase
                case = EvalCase(
                    question=result.question,
                    ground_truth_answer=result.answer,
                    ground_truth_contexts=[scenario.unit.content],
                    source_units=[source_unit],
                    persona=persona_dict,
                    generation_params={
                        "llm_uri": llm_uri,
                        "generated_at": datetime.now().isoformat(),
                        "case_index": index,
                        "theme": scenario.theme
                    }
                )
                return case
                
            except Exception as e:
                rprint(f"[yellow]⚠[/yellow] Failed to generate case {index+1}: {e}")
                return None
    
    # Concurrent generation
    rprint(f"[cyan]Generating {len(scenarios)} cases with max_concurrent={max_concurrent}...[/cyan]")
    tasks = [generate_one_case(scenario, i) for i, scenario in enumerate(scenarios)]
    results = await asyncio.gather(*tasks)
    
    # Filter out failed cases
    cases = [case for case in results if case is not None]
    failed_count = len(results) - len(cases)
    
    if failed_count > 0:
        rprint(f"[yellow]⚠[/yellow] Generation completed: {len(cases)}/{len(scenarios)} cases succeeded, {failed_count} failed")
    else:
        rprint(f"[green]✓[/green] Successfully generated all {len(cases)} cases")
    
    return EvalDataset(
        cases=cases,
        domain=domain,
        generated_at=datetime.now().isoformat()
    )
