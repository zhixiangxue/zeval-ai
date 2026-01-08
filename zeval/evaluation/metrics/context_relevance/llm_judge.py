"""Context Relevance metric using LLM-as-Judge with dual-judge evaluation"""

from typing import List
import chak
from pydantic import BaseModel, Field

from ..base import BaseMetric
from ....schemas.eval import EvalCase, EvalResult


class RelevanceRating(BaseModel):
    """LLM output: relevance rating"""
    rating: int = Field(description="Relevance rating: 0 (not relevant), 1 (partially relevant), 2 (fully relevant)")
    reason: str = Field(description="Brief explanation of the rating")


class ContextRelevance(BaseMetric):
    """
    Context Relevance metric for evaluating retrieved contexts relevance to question
    
    Evaluates whether the retrieved contexts are relevant to answering the question.
    Uses LLM-as-Judge with dual-judge approach for robust evaluation.
    
    Evaluation logic:
    1. Judge contexts relevance to question using two different prompts
    2. Average both judge ratings for final score
    3. Convert from 0-2 scale to 0.0-1.0 scale
    
    Applicable to: RAG, Agent, or any system with question + retrieved_contexts
    
    Score interpretation:
    - 1.0: Fully relevant (contexts contain information to answer the question)
    - 0.5: Partially relevant (contexts contain some related information)
    - 0.0: Not relevant (contexts do not help answer the question)
    
    Usage:
        metric = ContextRelevance(
            llm_uri="bailian/qwen-plus",
            api_key="your_api_key"
        )
        await metric.evaluate_batch(dataset.cases)
    """
    
    def __init__(self, llm_uri: str, api_key: str, name: str = "context_relevance"):
        """
        Initialize Context Relevance metric
        
        Args:
            llm_uri: LLM URI (e.g., 'bailian/qwen-plus')
            api_key: API key for the LLM
            name: Metric name (default: 'context_relevance')
        """
        super().__init__(name)
        self.llm_uri = llm_uri
        self.api_key = api_key
    
    async def _evaluate(self, case: EvalCase) -> EvalResult:
        """
        Evaluate context relevance for a single case
        
        Args:
            case: Case with question and retrieved_contexts
            
        Returns:
            EvalResult with context relevance score
        """
        # Validate required fields
        if not case.question:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No question to evaluate",
            )
        
        if not case.retrieved_contexts:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No retrieved contexts available",
            )
        
        # Join contexts
        context_text = "\n\n".join(case.retrieved_contexts)
        
        # Edge case: empty strings
        if not case.question.strip() or not context_text.strip():
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Question or context is empty",
            )
        
        # Edge case: question matches context exactly (likely invalid case)
        if case.question.strip() == context_text.strip():
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Question matches context exactly (invalid case)",
            )
        
        # Edge case: context is contained in question (invalid case)
        if context_text.strip() in case.question.strip():
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Context is contained in question (invalid case)",
            )
        
        # Get ratings from dual judges
        judge1_rating = await self._judge_relevance_1(case.question, context_text)
        judge2_rating = await self._judge_relevance_2(case.question, context_text)
        
        # Average the scores and convert to 0.0-1.0 scale
        if judge1_rating is not None and judge2_rating is not None:
            score = (judge1_rating.rating + judge2_rating.rating) / 2.0 / 2.0
            reason = f"Judge1: {judge1_rating.rating}/2 ({judge1_rating.reason}), Judge2: {judge2_rating.rating}/2 ({judge2_rating.reason})"
        elif judge1_rating is not None:
            score = judge1_rating.rating / 2.0
            reason = f"Only Judge1 available: {judge1_rating.rating}/2 ({judge1_rating.reason})"
        elif judge2_rating is not None:
            score = judge2_rating.rating / 2.0
            reason = f"Only Judge2 available: {judge2_rating.rating}/2 ({judge2_rating.reason})"
        else:
            score = 0.0
            reason = "Both judges failed to evaluate"
        
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            details={
                "judge1_rating": judge1_rating.rating if judge1_rating else None,
                "judge1_reason": judge1_rating.reason if judge1_rating else None,
                "judge2_rating": judge2_rating.rating if judge2_rating else None,
                "judge2_reason": judge2_rating.reason if judge2_rating else None,
            }
        )
    
    async def _judge_relevance_1(self, question: str, context: str) -> RelevanceRating | None:
        """
        First judge: Direct relevance evaluation
        
        Args:
            question: The user's question
            context: Retrieved contexts (joined)
            
        Returns:
            RelevanceRating or None if failed
        """
        prompt = f"""You are an expert evaluator designed to assess the relevance of retrieved contexts to a question.

Question:
{question}

Context:
{context}

Task:
Determine if the Context contains proper information to answer the Question.
Do NOT rely on your prior knowledge. Use ONLY what is in the Context and Question.

Rating scale:
- 0: Context does NOT contain any relevant information to answer the question
- 1: Context PARTIALLY contains relevant information to answer the question
- 2: Context contains RELEVANT information to answer the question

Provide:
1. rating: Your relevance score (0, 1, or 2)
2. reason: Brief explanation of your rating (1 sentence)
"""
        
        try:
            conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
            result = await conv.asend(prompt, returns=RelevanceRating)
            
            # Validate rating is in expected range
            if result and result.rating in [0, 1, 2]:
                return result
            else:
                print(f"Judge1 invalid rating: {result.rating if result else None}")
                return None
        except Exception as e:
            print(f"Judge1 evaluation failed: {e}")
            return None
    
    async def _judge_relevance_2(self, question: str, context: str) -> RelevanceRating | None:
        """
        Second judge: Alternative perspective for fairness
        
        Args:
            question: The user's question
            context: Retrieved contexts (joined)
            
        Returns:
            RelevanceRating or None if failed
        """
        prompt = f"""As an expert designed to assess relevance, determine the extent to which the given Context provides information necessary to answer the Question.

Rely SOLELY on the information in the Context and Question, not on prior knowledge.

Question:
{question}

Context:
{context}

Instructions:
- If the Context does NOT contain any relevant information → rating = 0
- If the Context PARTIALLY contains relevant information → rating = 1  
- If the Context contains RELEVANT information → rating = 2

Provide:
1. rating: Your relevance score (0, 1, or 2)
2. reason: Brief explanation (1 sentence)
"""
        
        try:
            conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
            result = await conv.asend(prompt, returns=RelevanceRating)
            
            # Validate rating is in expected range
            if result and result.rating in [0, 1, 2]:
                return result
            else:
                print(f"Judge2 invalid rating: {result.rating if result else None}")
                return None
        except Exception as e:
            print(f"Judge2 evaluation failed: {e}")
            return None
