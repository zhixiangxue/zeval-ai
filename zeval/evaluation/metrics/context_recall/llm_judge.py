"""Context Recall metric using LLM-as-Judge approach"""

import chak
from typing import List
from pydantic import BaseModel, Field

from ..base import BaseMetric
from ....schemas.eval import EvalCase, EvalResult


class StatementClassification(BaseModel):
    """Classification result for a single statement"""
    statement: str = Field(description="The statement text")
    reason: str = Field(description="Reason for the attribution judgment")
    attributed: int = Field(description="1 if attributed to context, 0 otherwise")


class StatementClassifications(BaseModel):
    """Collection of statement classifications"""
    classifications: List[StatementClassification] = Field(
        description="List of statement classifications"
    )


CLASSIFICATION_PROMPT = """Given a context and an answer, analyze each statement in the answer and classify if the statement can be attributed to the given context or not.

Use only binary classification: 1 if the statement can be attributed to the context, 0 if it cannot.
Provide detailed reasoning for each classification.

Question: {question}

Context:
{context}

Answer to analyze:
{answer}

For each statement in the answer, determine:
1. Extract the statement
2. Check if it can be supported/verified by the context
3. Provide reasoning
4. Assign 1 (attributed) or 0 (not attributed)

Analyze carefully and be strict - only mark as attributed (1) if the context clearly supports the statement."""


class ContextRecall(BaseMetric):
    """
    Context Recall metric for evaluating retrieval completeness
    
    Evaluates whether the retrieved contexts contain all information needed 
    to answer the question. Uses ground_truth_answer as reference.
    
    Algorithm:
    1. Decompose ground_truth_answer into atomic statements
    2. For each statement, check if it can be attributed to retrieved_contexts
    3. Calculate recall = attributed_statements / total_statements
    
    Applicable to: RAG, Agent, or any system with retrieval + ground truth
    """
    
    def __init__(self, llm_uri: str, api_key: str, name: str = "context_recall"):
        """
        Initialize Context Recall metric
        
        Args:
            llm_uri: LLM endpoint URI for chak
            api_key: API key for the LLM
            name: Metric name (default: "context_recall")
        """
        super().__init__(name)
        self.llm_uri = llm_uri
        self.api_key = api_key
    
    async def _evaluate(self, case: EvalCase) -> EvalResult:
        """
        Evaluate context recall for a single case
        
        Args:
            case: Evaluation case with question, ground_truth_answer, and retrieved_contexts
            
        Returns:
            EvalResult with recall score (0.0-1.0)
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
        
        # Prepare context text
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{ctx}"
            for i, ctx in enumerate(case.retrieved_contexts)
        ])
        
        # Call LLM to classify statements
        classifications = await self._classify_statements(
            question=case.question,
            context=context_text,
            answer=case.ground_truth_answer
        )
        
        if not classifications:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Failed to classify statements",
            )
        
        # Calculate recall score
        total = len(classifications)
        attributed = sum(1 for c in classifications if c.attributed == 1)
        score = attributed / total if total > 0 else 0.0
        
        # Build detailed reason
        details = []
        for c in classifications:
            status = "✓" if c.attributed == 1 else "✗"
            details.append(f"{status} {c.statement[:100]}... - {c.reason}")
        
        reason = f"Recall: {attributed}/{total} statements attributed. " + "; ".join(details[:3])
        if len(details) > 3:
            reason += f" ... (and {len(details)-3} more)"
        
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            details={
                "total_statements": total,
                "attributed_statements": attributed,
                "classifications": [
                    {
                        "statement": c.statement,
                        "reason": c.reason,
                        "attributed": c.attributed
                    }
                    for c in classifications
                ]
            }
        )
    
    async def _classify_statements(
        self,
        question: str,
        context: str,
        answer: str
    ) -> List[StatementClassification]:
        """
        Classify each statement in the answer as attributed or not
        
        Args:
            question: The original question
            context: Retrieved contexts (concatenated)
            answer: Ground truth answer to analyze
            
        Returns:
            List of statement classifications
        """
        # Create new conversation to avoid state pollution
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        prompt = CLASSIFICATION_PROMPT.format(
            question=question,
            context=context,
            answer=answer
        )
        
        try:
            result = await conv.asend(prompt, returns=StatementClassifications)
            return result.classifications if result and result.classifications else []
        except Exception as e:
            print(f"[ERROR] Failed to classify statements: {e}")
            return []
