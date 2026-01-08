"""Context Precision metric using LLM-as-Judge approach"""

import chak
from typing import List
from pydantic import BaseModel, Field

from ..base import BaseMetric
from ....schemas.eval import EvalCase, EvalResult


class ContextVerdict(BaseModel):
    """Verdict for a single context's usefulness"""
    reason: str = Field(description="Reason for the verdict")
    verdict: int = Field(description="1 if useful, 0 if not useful")


VERIFICATION_PROMPT = """Given a question, an answer, and a context, verify if the context was useful in arriving at the given answer.

Question: {question}

Answer: {answer}

Context to evaluate:
{context}

Determine if this context was useful for answering the question. Consider:
1. Does the context contain information that directly supports the answer?
2. Does the context provide relevant background or details mentioned in the answer?
3. Would removing this context make the answer harder to arrive at?

Give verdict as "1" if the context was useful and "0" if it was not useful.
Provide a clear reason for your verdict."""


class ContextPrecision(BaseMetric):
    """
    Context Precision metric for evaluating retrieval ranking quality
    
    Evaluates whether retrieved contexts are useful and properly ranked.
    Uses ground_truth_answer (or ground_truth_contexts) as reference.
    
    Algorithm:
    1. For each retrieved context, judge if it's useful for answering the question
    2. Calculate Average Precision based on verdicts
    3. Average Precision rewards relevant contexts appearing earlier in the list
    
    Formula:
    AP = (Σ(P@k × rel(k))) / total_relevant
    where P@k = precision at position k, rel(k) = relevance at position k
    
    Applicable to: RAG, Agent, or any system with retrieval + ground truth
    """
    
    def __init__(self, llm_uri: str, api_key: str, name: str = "context_precision"):
        """
        Initialize Context Precision metric
        
        Args:
            llm_uri: LLM endpoint URI for chak
            api_key: API key for the LLM
            name: Metric name (default: "context_precision")
        """
        super().__init__(name)
        self.llm_uri = llm_uri
        self.api_key = api_key
    
    async def _evaluate(self, case: EvalCase) -> EvalResult:
        """
        Evaluate context precision for a single case
        
        Args:
            case: Evaluation case with question, ground_truth_answer, and retrieved_contexts
            
        Returns:
            EvalResult with precision score (0.0-1.0)
        """
        # Validate required fields
        if not case.ground_truth_answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No ground_truth_answer provided",
            )
        
        if not case.retrieved_contexts:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No retrieved_contexts provided",
            )
        
        # Evaluate each retrieved context
        verdicts = []
        verdict_details = []
        
        for i, context in enumerate(case.retrieved_contexts):
            verdict = await self._verify_context_usefulness(
                question=case.question,
                answer=case.ground_truth_answer,
                context=context
            )
            
            if verdict:
                verdicts.append(verdict.verdict)
                verdict_details.append({
                    "position": i + 1,
                    "verdict": verdict.verdict,
                    "reason": verdict.reason,
                    "context_preview": context[:100] + "..." if len(context) > 100 else context
                })
            else:
                # If LLM fails, assume not useful
                verdicts.append(0)
                verdict_details.append({
                    "position": i + 1,
                    "verdict": 0,
                    "reason": "Failed to get verdict",
                    "context_preview": context[:100] + "..." if len(context) > 100 else context
                })
        
        # Calculate Average Precision
        score = self._calculate_average_precision(verdicts)
        
        # Build reason
        useful_count = sum(verdicts)
        total_count = len(verdicts)
        
        # Show verdict summary
        verdict_summary = []
        for detail in verdict_details[:3]:  # Show first 3
            status = "✓" if detail["verdict"] == 1 else "✗"
            verdict_summary.append(
                f"[{detail['position']}] {status} {detail['context_preview']}"
            )
        
        reason = f"Average Precision: {score:.2f}. Useful contexts: {useful_count}/{total_count}. "
        reason += "; ".join(verdict_summary)
        if len(verdict_details) > 3:
            reason += f" ... (and {len(verdict_details)-3} more)"
        
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            details={
                "verdicts": verdicts,
                "useful_count": useful_count,
                "total_count": total_count,
                "verdict_details": verdict_details,
                "average_precision": score
            }
        )
    
    async def _verify_context_usefulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> ContextVerdict:
        """
        Verify if a context is useful for answering the question
        
        Args:
            question: The original question
            answer: Ground truth answer (or reference answer)
            context: Context to verify
            
        Returns:
            ContextVerdict with verdict (1=useful, 0=not useful) and reason
        """
        # Create new conversation to avoid state pollution
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        prompt = VERIFICATION_PROMPT.format(
            question=question,
            answer=answer,
            context=context
        )
        
        try:
            result = await conv.asend(prompt, returns=ContextVerdict)
            return result
        except Exception as e:
            print(f"[ERROR] Failed to verify context usefulness: {e}")
            return None
    
    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        """
        Calculate Average Precision from binary verdicts
        
        Average Precision rewards relevant items appearing earlier in the ranked list.
        
        Formula:
        AP = (Σ(P@k × rel(k))) / total_relevant
        where:
        - P@k = precision at position k = (relevant items in top k) / k
        - rel(k) = relevance at position k (1 if relevant, 0 otherwise)
        
        Args:
            verdicts: List of binary verdicts (1=useful, 0=not useful) in order
            
        Returns:
            Average Precision score in [0.0, 1.0]
        """
        if not verdicts:
            return 0.0
        
        total_relevant = sum(verdicts)
        
        # If no relevant items, return 0
        if total_relevant == 0:
            return 0.0
        
        # Calculate sum of (Precision@k × relevance@k)
        precision_sum = 0.0
        for i, verdict in enumerate(verdicts):
            if verdict == 1:
                # Precision at position i+1 = (relevant items in top i+1) / (i+1)
                precision_at_k = sum(verdicts[:i+1]) / (i + 1)
                precision_sum += precision_at_k
        
        # Average Precision
        average_precision = precision_sum / total_relevant
        
        return float(average_precision)
