"""
Summary extractor
"""

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import BaseExtractor
from ....schemas.base import BaseUnit


class SummarySchema(BaseModel):
    """Summary output schema"""
    summary: str


class SummaryExtractor(BaseExtractor):
    """
    Summary extractor
    
    Input: unit.content (text content)
    Prompt: "Summarize in N sentences" with few-shot examples
    Output: Populates unit.summary
    
    Attributes:
        model_uri: Model URI
        api_key: API key
        max_sentences: Maximum number of sentences in summary (default: 2)
    
    Example:
        extractor = SummaryExtractor(
            model_uri="openai/gpt-4o-mini",
            api_key="sk-xxx",
            max_sentences=2
        )
        await extractor.extract(unit)  # populates unit.summary
    """
    
    def __init__(self, model_uri: str, api_key: str, max_sentences: int = 2):
        super().__init__(model_uri, api_key)
        self.max_sentences = max_sentences
    
    def build_prompt(self, unit: BaseUnit) -> str:
        """Build summary prompt with few-shot examples"""
        return f"""Summarize the given text in less than {self.max_sentences} sentences.

# Example 1
Input:
Artificial intelligence

Artificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.

Output:
AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations.

# Your Task
Input:
{unit.content}

Output:"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def extract(self, unit: BaseUnit) -> bool:
        """Extract summary and update unit.summary"""
        try:
            prompt = self.build_prompt(unit)
            result = await self.conv.asend(prompt, returns=SummarySchema)
            unit.summary = result.summary
            return True
        except Exception as e:
            print(f"[SKIP] summary extraction failed for unit {unit.unit_id}: {e}")
            return False
