"""
Named entities extractor
"""

from typing import List
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import BaseExtractor
from ....schemas.base import BaseUnit


class EntitiesSchema(BaseModel):
    """Entities output schema"""
    entities: List[str]


class EntitiesExtractor(BaseExtractor):
    """
    Named entities extractor
    
    Input: unit.content (text content)
    Prompt: "Extract named entities" with few-shot examples
    Output: Populates unit.entities
    
    Attributes:
        model_uri: Model URI
        api_key: API key
        max_num: Maximum number of entities to extract (default: 10)
    
    Example:
        extractor = EntitiesExtractor(
            model_uri="openai/gpt-4o-mini",
            api_key="sk-xxx",
            max_num=10
        )
        await extractor.extract(unit)  # populates unit.entities
    """
    
    def __init__(self, model_uri: str, api_key: str, max_num: int = 10):
        super().__init__(model_uri, api_key)
        self.max_num = max_num
    
    def build_prompt(self, unit: BaseUnit) -> str:
        """Build entities extraction prompt with few-shot examples"""
        return f"""Extract the named entities from the given text, limiting the output to the top {self.max_num} entities.
Focus on people, organizations, locations, dates, and other proper nouns.

# Example 1
Input:
Elon Musk, the CEO of Tesla and SpaceX, announced plans to expand operations to new locations in Europe and Asia. This expansion is expected to create thousands of jobs, particularly in cities like Berlin and Shanghai.

Max entities: 10

Output:
- Elon Musk
- Tesla
- SpaceX
- Europe
- Asia
- Berlin
- Shanghai

# Your Task
Input:
{unit.content}

Max entities: {self.max_num}

Output:"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def extract(self, unit: BaseUnit) -> bool:
        """Extract entities and update unit.entities"""
        try:
            prompt = self.build_prompt(unit)
            conv = self.create_conv()  # Create new conversation for each extraction
            result = await conv.asend(prompt, returns=EntitiesSchema)
            unit.entities = result.entities
            return True
        except Exception as e:
            print(f"[SKIP] entities extraction failed for unit {unit.unit_id}: {e}")
            return False
