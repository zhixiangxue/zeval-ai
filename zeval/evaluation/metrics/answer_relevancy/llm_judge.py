"""Answer Relevancy metric using LLM-as-Judge approach"""

import chak
from typing import List
from pydantic import BaseModel, Field

from ..base import BaseMetric
from ....schemas.eval import EvalCase, EvalResult


class RelevancyJudgment(BaseModel):
    """Judgment of answer relevancy"""
    score: float = Field(description="Relevancy score from 0.0 to 1.0")
    reason: str = Field(description="Reason for the score")
    is_direct: bool = Field(description="Whether the answer directly addresses the question")
    has_irrelevant_info: bool = Field(description="Whether the answer contains irrelevant information")
    is_noncommittal: bool = Field(description="Whether the answer is evasive or vague")


RELEVANCY_PROMPT = """Evaluate how relevant the answer is to the question. Consider the following criteria:

1. **Directness**: Does the answer directly address the question?
2. **Focus**: Does the answer stay on topic without including irrelevant information?
3. **Commitment**: Is the answer substantive, or is it evasive/vague (e.g., "I don't know", "It depends")?

Question: {question}

Answer: {answer}

Scoring Guidelines:
- Score 1.0 (Perfect): Answer directly addresses the question, stays focused, and is substantive
- Score 0.7-0.9 (Good): Answer addresses the question but may include some extra context or minor tangential info
- Score 0.4-0.6 (Moderate): Answer partially addresses the question but includes significant irrelevant information or is somewhat indirect
- Score 0.1-0.3 (Poor): Answer barely addresses the question, mostly irrelevant content
- Score 0.0 (Irrelevant): Answer is completely off-topic, evasive, or noncommittal

Provide:
1. A score from 0.0 to 1.0
2. A clear reason for the score
3. Whether the answer is direct (true/false)
4. Whether the answer contains irrelevant information (true/false)
5. Whether the answer is noncommittal/evasive (true/false)"""


class AnswerRelevancy(BaseMetric):
    """
    Answer Relevancy metric for evaluating if answer addresses the question
    
    Evaluates whether the answer is relevant, focused, and substantive.
    Does NOT evaluate correctness - only relevancy to the question.
    
    Evaluation criteria:
    1. Directness: Does the answer directly address the question?
    2. Focus: Does the answer avoid irrelevant information?
    3. Commitment: Is the answer substantive (not evasive/vague)?
    
    Applicable to: Any QA system (RAG, Agent, chatbot, etc.)
    """
    
    def __init__(self, llm_uri: str, api_key: str, name: str = "answer_relevancy"):
        """
        Initialize Answer Relevancy metric
        
        Args:
            llm_uri: LLM endpoint URI for chak
            api_key: API key for the LLM
            name: Metric name (default: "answer_relevancy")
        """
        super().__init__(name)
        self.llm_uri = llm_uri
        self.api_key = api_key
    
    async def _evaluate(self, case: EvalCase) -> EvalResult:
        """
        Evaluate answer relevancy for a single case
        
        Args:
            case: Evaluation case with question and answer
            
        Returns:
            EvalResult with relevancy score (0.0-1.0)
        """
        # Validate required fields
        if not case.answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No answer provided",
            )
        
        if not case.question:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No question provided",
            )
        
        # Judge answer relevancy
        judgment = await self._judge_relevancy(
            question=case.question,
            answer=case.answer
        )
        
        if not judgment:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Failed to judge relevancy",
            )
        
        # Build detailed reason
        criteria = []
        if judgment.is_direct:
            criteria.append("✓ Direct")
        else:
            criteria.append("✗ Indirect")
        
        if not judgment.has_irrelevant_info:
            criteria.append("✓ Focused")
        else:
            criteria.append("✗ Has irrelevant info")
        
        if not judgment.is_noncommittal:
            criteria.append("✓ Substantive")
        else:
            criteria.append("✗ Evasive/vague")
        
        criteria_summary = ", ".join(criteria)
        full_reason = f"{criteria_summary}. {judgment.reason}"
        
        return EvalResult(
            metric_name=self.name,
            score=judgment.score,
            reason=full_reason,
            details={
                "is_direct": judgment.is_direct,
                "has_irrelevant_info": judgment.has_irrelevant_info,
                "is_noncommittal": judgment.is_noncommittal,
                "judge_reason": judgment.reason
            }
        )
    
    async def _judge_relevancy(
        self,
        question: str,
        answer: str
    ) -> RelevancyJudgment:
        """
        Judge the relevancy of an answer to a question
        
        Args:
            question: The original question
            answer: The answer to evaluate
            
        Returns:
            RelevancyJudgment with score, reason, and flags
        """
        # Create new conversation to avoid state pollution
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        prompt = RELEVANCY_PROMPT.format(
            question=question,
            answer=answer
        )
        
        try:
            result = await conv.asend(prompt, returns=RelevancyJudgment)
            
            # Ensure score is in valid range
            if result:
                result.score = max(0.0, min(1.0, result.score))
            
            return result
        except Exception as e:
            print(f"[ERROR] Failed to judge relevancy: {e}")
            return None
