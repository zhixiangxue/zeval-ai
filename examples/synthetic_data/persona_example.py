"""
Test persona generation with custom fields
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic import Field
from zeval.synthetic_data.generators.persona import Persona, generate_personas

# Load environment variables
load_dotenv()


# Define custom persona with domain-specific fields
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
    """Test persona generation"""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    print("\n" + "="*60)
    print("Testing Custom Persona Generation")
    print("="*60)
    
    # Test: US Home Buyers with custom fields
    print("\n[Test] Domain: US Home Buyers (with financial attributes)")
    print("-" * 60)
    
    personas = await generate_personas(
        llm_uri="openai/gpt-4o-mini",
        api_key=api_key,
        domain="US residential real estate and home buying process",
        num_personas=5,
        persona_model=HomeBuyerPersona  # Use custom persona model
    )
    
    for i, persona in enumerate(personas, 1):
        print(f"\n{i}. {persona.name}")
        print(f"   Expertise: {persona.expertise_level}")
        print(f"   Role: {persona.role_description}")
        print(f"   Focus: {persona.focus_area}")
        print(f"   💳 Credit Score: {persona.credit_score}")
        print(f"   📊 DTI Ratio: {persona.dti_ratio}%")
        print(f"   💰 Down Payment: {persona.down_payment_percent}%")
        print(f"   🏠 Budget: {persona.budget_range}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
