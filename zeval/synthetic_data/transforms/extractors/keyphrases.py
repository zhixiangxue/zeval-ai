"""
Keyphrases extractor
"""

from typing import List
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import BaseExtractor
from ....schemas.base import BaseUnit


class KeyphrasesSchema(BaseModel):
    """Keyphrases output schema"""
    keyphrases: List[str]


class KeyphrasesExtractor(BaseExtractor):
    """
    Keyphrases extractor
    
    Input: unit.content (text content)
    Prompt: "Extract top N keyphrases" with few-shot examples
    Output: Populates unit.keyphrases
    
    Attributes:
        model_uri: Model URI
        api_key: API key
        max_num: Maximum number of keyphrases to extract (default: 5)
    
    Example:
        extractor = KeyphrasesExtractor(
            model_uri="openai/gpt-4o-mini",
            api_key="sk-xxx",
            max_num=5
        )
        await extractor.extract(unit)  # populates unit.keyphrases
    """
    
    def __init__(self, model_uri: str, api_key: str, max_num: int = 5):
        super().__init__(model_uri, api_key)
        self.max_num = max_num
    
    def build_prompt(self, unit: BaseUnit) -> str:
        """Build keyphrases extraction prompt with few-shot examples"""
        return f"""Extract the top {self.max_num} keyphrases from the given text.

# Example 1
Input:
Artificial intelligence

Artificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.

Max keyphrases: 5

Output:
- Artificial intelligence
- automating tasks
- healthcare
- self-driving cars
- personalized recommendations

# Your Task
Input:
{unit.content}

Max keyphrases: {self.max_num}

Output:"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def extract(self, unit: BaseUnit) -> bool:
        """Extract keyphrases and update unit.keyphrases"""
        try:
            prompt = self.build_prompt(unit)
            conv = self.create_conv()  # Create new conversation for each extraction
            result = await conv.asend(prompt, returns=KeyphrasesSchema)
            unit.keyphrases = result.keyphrases
            return True
        except Exception as e:
            print(f"[SKIP] keyphrases extraction failed for unit {unit.unit_id}: {e}")
            return False
