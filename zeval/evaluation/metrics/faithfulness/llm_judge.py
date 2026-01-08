"""Faithfulness metric using LLM-as-Judge with statement decomposition"""

from typing import List
import chak
from pydantic import BaseModel, Field

from ..base import BaseMetric, Statement
from ....schemas.eval import EvalCase, EvalResult


class DecomposedStatements(BaseModel):
    """LLM output: decomposed atomic statements"""
    statements: List[str] = Field(description="List of atomic statements")


class Faithfulness(BaseMetric):
    """
    Faithfulness metric for evaluating answer faithfulness to retrieved contexts
    
    Evaluates whether the generated answer is faithful to the retrieved contexts.
    Uses LLM-as-Judge approach:
    1. Decompose answer into atomic statements
    2. Verify each statement against retrieved contexts using NLI
    3. Calculate faithfulness score = supported statements / total statements
    
    Applicable to: RAG, Agent, or any system with answer + contexts
    
    Score interpretation:
    - 1.0: Fully faithful (all statements supported)
    - 0.8-1.0: Mostly faithful (minor unsupported details)
    - 0.5-0.8: Partially faithful (some hallucinations)
    - 0.0-0.5: Low faithfulness (significant hallucinations)
    
    Usage:
        metric = Faithfulness(
            llm_uri="bailian/qwen-plus",
            api_key="your_api_key"
        )
        await metric.evaluate_batch(dataset.cases)
    """
    
    def __init__(self, llm_uri: str, api_key: str, name: str = "faithfulness"):
        """
        Initialize Faithfulness metric
        
        Args:
            llm_uri: LLM URI (e.g., 'bailian/qwen-plus')
            api_key: API key for the LLM
            name: Metric name (default: 'faithfulness')
        """
        super().__init__(name)
        self.llm_uri = llm_uri
        self.api_key = api_key
    
    async def _evaluate(self, case: EvalCase) -> EvalResult:
        """
        Evaluate faithfulness for a single case
        
        Args:
            case: Case with answer and retrieved_contexts
            
        Returns:
            EvalResult with faithfulness score
        """
        # Validate required fields
        if not case.answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No answer to evaluate (answer is None or empty)",
            )
        
        if not case.retrieved_contexts:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No retrieved contexts available",
            )
        
        # Step 1: Decompose answer into statements
        statements = await self._decompose_statements(case.answer)
        
        if not statements:
            # If no statements extracted, consider it fully faithful (empty is vacuously true)
            return EvalResult(
                metric_name=self.name,
                score=1.0,
                reason="No statements extracted from answer",
                details={"statements": []},
            )
        
        # Step 2: Verify each statement against contexts
        verified_statements = await self._verify_statements(
            statements, 
            case.retrieved_contexts
        )
        
        # Step 3: Calculate score
        supported_count = sum(1 for s in verified_statements if s.supported)
        total_count = len(verified_statements)
        score = supported_count / total_count if total_count > 0 else 0.0
        
        # Build reason
        reason = f"{supported_count}/{total_count} statements supported by context"
        
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            details={
                "statements": [s.model_dump() for s in verified_statements],
                "supported_count": supported_count,
                "total_count": total_count,
            }
        )
    
    async def _decompose_statements(self, text: str) -> List[str]:
        """
        Decompose text into atomic statements using LLM
        
        Args:
            text: Text to decompose (e.g., answer)
            
        Returns:
            List of atomic statement strings
        """
        prompt = f"""Decompose the following text into atomic statements. Each statement should be an independent, verifiable fact.

Text:
{text}

Requirements:
1. Each statement contains only one fact
2. Statements should be complete and independent
3. Preserve the original semantics
4. Do not add information not present in the original text
"""
        
        try:
            # Create new conversation for each call to avoid state pollution
            conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
            result = await conv.asend(prompt, returns=DecomposedStatements)
            return result.statements if result and result.statements else []
        except Exception as e:
            print(f"Statement decomposition failed: {e}")
            return []
    
    async def _verify_statements(
        self, 
        statements: List[str], 
        contexts: List[str]
    ) -> List[Statement]:
        """
        Verify statements against contexts using NLI
        
        Args:
            statements: List of atomic statements
            contexts: Retrieved contexts
            
        Returns:
            List of Statement objects with support status
        """
        context_text = "\n\n".join(contexts)
        
        prompt = f"""Judge whether each statement is supported by the context.

Context:
{context_text}

Statements:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(statements))}

Requirements:
- If the statement can be directly inferred or derived from the context, mark supported=true
- If the statement contains information not in the context, mark supported=false

For each statement, provide:
- text: the original statement
- supported: true or false
"""
        
        # Create a wrapper model for the list
        class VerifiedStatements(BaseModel):
            statements: List[Statement] = Field(description="List of verified statements")
        
        try:
            # Create new conversation for each call to avoid state pollution
            conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
            result = await conv.asend(prompt, returns=VerifiedStatements)
            return result.statements if result and result.statements else []
        except Exception as e:
            print(f"Statement verification failed: {e}")
            # On failure, mark all as unsupported
            return [Statement(text=s, supported=False) for s in statements]
