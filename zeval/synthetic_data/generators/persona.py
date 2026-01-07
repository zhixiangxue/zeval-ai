"""
Persona generator for test data synthesis

Simplified approach compared to Ragas: directly use LLM to generate
diverse user personas based on domain description, without complex
embedding clustering.
"""

from pydantic import BaseModel, Field
from typing import Type
import chak
from tenacity import retry, stop_after_attempt, wait_exponential


class Persona(BaseModel):
    """
    User persona for test generation
    
    Represents a type of user who would interact with the content.
    Used to generate questions/tests tailored to different user profiles.
    
    Attributes:
        name: Descriptive role name (e.g., "ML Beginner", "Data Scientist")
        role_description: What they do and their goals (1-2 sentences)
        expertise_level: Their skill level ("beginner" | "intermediate" | "expert")
        focus_area: Their specific area of interest within the domain
    """
    name: str = Field(description="Descriptive role name")
    role_description: str = Field(description="What they do and their goals")
    expertise_level: str = Field(
        description="Skill level: beginner, intermediate, or expert"
    )
    focus_area: str = Field(description="Specific area of interest")


class PersonaList(BaseModel):
    """List of personas"""
    personas: list[Persona] = Field(description="List of user personas")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def generate_personas(
    llm_uri: str,
    api_key: str,
    domain: str,
    num_personas: int = 3,
    persona_model: Type[Persona] = Persona
) -> list[Persona]:
    """
    Generate diverse user personas for a given domain using LLM
    
    Flexible approach: Accept custom Persona model with additional fields.
    Field descriptions are automatically extracted and used in prompt.
    
    Args:
        llm_uri: LLM URI (e.g., "openai/gpt-4o-mini")
        api_key: API key for the LLM
        domain: Domain description (e.g., "US residential real estate")
        num_personas: Number of personas to generate (default: 3)
        persona_model: Pydantic model class inheriting from Persona (default: Persona)
                      Custom models can add domain-specific fields with descriptions
    
    Returns:
        List of persona instances (of type persona_model)
    
    Example 1 - Basic usage:
        >>> personas = generate_personas(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-...",
        ...     domain="machine learning",
        ...     num_personas=3
        ... )
    
    Example 2 - Custom persona with domain-specific fields:
        >>> class HomeBuyerPersona(Persona):
        ...     credit_score: int = Field(
        ...         description="Credit score (300-850), affects loan eligibility"
        ...     )
        ...     dti_ratio: float = Field(
        ...         description="Debt-to-Income ratio (%), max 43%"
        ...     )
        ...
        >>> personas = generate_personas(
        ...     llm_uri="openai/gpt-4o-mini",
        ...     api_key="sk-...",
        ...     domain="US residential real estate",
        ...     num_personas=5,
        ...     persona_model=HomeBuyerPersona
        ... )
        >>> # Generated personas will include credit_score and dti_ratio
    """
    import inspect
    
    conv = chak.Conversation(llm_uri, api_key=api_key)
    
    # Extract field descriptions from persona_model
    field_descriptions = []
    for field_name, field_info in persona_model.model_fields.items():
        desc = field_info.description or "(no description)"
        field_type = field_info.annotation
        # Format type nicely
        if hasattr(field_type, '__name__'):
            type_str = field_type.__name__
        else:
            type_str = str(field_type).replace('typing.', '')
        field_descriptions.append(f"- {field_name} ({type_str}): {desc}")
    
    fields_prompt = "\n".join(field_descriptions)
    
    prompt = f"""Generate {num_personas} diverse user personas for the domain: {domain}

Each persona should represent a different user type who would interact with this content.

Requirements:
1. Vary expertise levels across: beginner, intermediate, expert
2. Cover different focus areas within the domain
3. Represent realistic user types with distinct needs and goals
4. Make personas diverse and non-overlapping

For each persona, provide ALL of the following fields:
{fields_prompt}

IMPORTANT: 
- For numeric fields (credit_score, dti_ratio, etc.), provide realistic values based on the persona's profile
- Ensure each persona has internally consistent characteristics
- Generate exactly {num_personas} distinct personas"""
    
    # Create a list wrapper for returning multiple personas
    class PersonaListWrapper(BaseModel):
        personas: list[persona_model] = Field(description="List of user personas")
    
    result = await conv.asend(prompt, returns=PersonaListWrapper)
    return result.personas
