"""Answer Correctness metric using LLM-as-Judge approach"""

import chak
from typing import List
from pydantic import BaseModel, Field

from ..base import BaseMetric
from ....schemas.eval import EvalCase, EvalResult


class StatementClassification(BaseModel):
    """Classification of a single statement"""
    statement: str = Field(description="The statement being classified")
    category: str = Field(description="TP, FP, or FN")
    reason: str = Field(description="Reason for the classification")


class CorrectnessClassification(BaseModel):
    """Complete classification result"""
    TP: List[StatementClassification] = Field(description="True Positive statements")
    FP: List[StatementClassification] = Field(description="False Positive statements")
    FN: List[StatementClassification] = Field(description="False Negative statements")


class DecomposedStatements(BaseModel):
    """Decomposed atomic statements"""
    statements: List[str] = Field(description="List of atomic statements")


DECOMPOSITION_PROMPT = """Given a question and an answer, break down the answer into atomic statements. Each statement should be:
1. A single, independent fact or claim
2. Fully understandable on its own (no pronouns like "he", "she", "it")
3. Simple and clear

Question: {question}

Answer: {answer}

Break down this answer into atomic statements. Each statement should stand alone and be verifiable."""


CLASSIFICATION_PROMPT = """Given a question, analyze the answer statements and ground truth statements. Classify each statement into one of three categories:

**TP (True Positive)**: Statements in the answer that are directly supported by the ground truth
**FP (False Positive)**: Statements in the answer that are NOT supported by the ground truth (incorrect or unsupported claims)
**FN (False Negative)**: Statements in the ground truth that are missing from the answer

Question: {question}

Answer Statements:
{answer_statements}

Ground Truth Statements:
{ground_truth_statements}

For each statement, determine:
1. Which category it belongs to (TP, FP, or FN)
2. A clear reason for the classification

Rules:
- Each statement can only belong to ONE category
- Be strict: TP requires direct support from ground truth
- FP includes incorrect facts or unsupported claims
- FN includes important information from ground truth missing in answer"""


class AnswerCorrectness(BaseMetric):
    """
    Answer Correctness metric for evaluating factual accuracy
    
    Evaluates whether the answer is factually correct compared to ground truth.
    Uses statement-level TP/FP/FN classification to compute F1 score.
    
    Algorithm:
    1. Decompose both answer and ground_truth_answer into atomic statements
    2. Classify each statement as TP (correct), FP (incorrect), or FN (missing)
    3. Calculate F1 score: F1 = 2*TP / (2*TP + FP + FN)
    
    Key metrics:
    - Precision: TP / (TP + FP) - accuracy of provided information
    - Recall: TP / (TP + FN) - completeness of answer
    - F1: Harmonic mean of precision and recall
    
    Applicable to: Any QA system with ground truth answers
    """
    
    def __init__(self, llm_uri: str, api_key: str, name: str = "answer_correctness"):
        """
        Initialize Answer Correctness metric
        
        Args:
            llm_uri: LLM endpoint URI for chak
            api_key: API key for the LLM
            name: Metric name (default: "answer_correctness")
        """
        super().__init__(name)
        self.llm_uri = llm_uri
        self.api_key = api_key
    
    async def _evaluate(self, case: EvalCase) -> EvalResult:
        """
        Evaluate answer correctness for a single case
        
        Args:
            case: Evaluation case with question, answer, and ground_truth_answer
            
        Returns:
            EvalResult with correctness score (0.0-1.0)
        """
        # Validate required fields
        if not case.answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No answer provided",
            )
        
        if not case.ground_truth_answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No ground_truth_answer provided",
            )
        
        # Step 1: Decompose answer into statements
        answer_statements = await self._decompose_statements(
            question=case.question,
            answer=case.answer
        )
        
        if not answer_statements:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Failed to decompose answer into statements",
            )
        
        # Step 2: Decompose ground truth into statements
        ground_truth_statements = await self._decompose_statements(
            question=case.question,
            answer=case.ground_truth_answer
        )
        
        if not ground_truth_statements:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Failed to decompose ground truth into statements",
            )
        
        # Step 3: Classify statements as TP/FP/FN
        classification = await self._classify_statements(
            question=case.question,
            answer_statements=answer_statements,
            ground_truth_statements=ground_truth_statements
        )
        
        if not classification:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="Failed to classify statements",
            )
        
        # Step 4: Calculate F1 score
        tp = len(classification.TP)
        fp = len(classification.FP)
        fn = len(classification.FN)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
        
        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Build detailed reason
        reason_parts = [
            f"F1={f1_score:.2f} (P={precision:.2f}, R={recall:.2f})",
            f"TP={tp}, FP={fp}, FN={fn}"
        ]
        
        # Add examples
        if classification.TP:
            tp_example = classification.TP[0].statement[:80] + "..." if len(classification.TP[0].statement) > 80 else classification.TP[0].statement
            reason_parts.append(f"✓ Correct: {tp_example}")
        
        if classification.FP:
            fp_example = classification.FP[0].statement[:80] + "..." if len(classification.FP[0].statement) > 80 else classification.FP[0].statement
            reason_parts.append(f"✗ Incorrect: {fp_example}")
        
        if classification.FN:
            fn_example = classification.FN[0].statement[:80] + "..." if len(classification.FN[0].statement) > 80 else classification.FN[0].statement
            reason_parts.append(f"⚠ Missing: {fn_example}")
        
        reason = "; ".join(reason_parts)
        
        return EvalResult(
            metric_name=self.name,
            score=f1_score,
            reason=reason,
            details={
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "tp_count": tp,
                "fp_count": fp,
                "fn_count": fn,
                "TP": [{"statement": s.statement, "reason": s.reason} for s in classification.TP],
                "FP": [{"statement": s.statement, "reason": s.reason} for s in classification.FP],
                "FN": [{"statement": s.statement, "reason": s.reason} for s in classification.FN],
            }
        )
    
    async def _decompose_statements(self, question: str, answer: str) -> List[str]:
        """
        Decompose an answer into atomic statements
        
        Args:
            question: The original question
            answer: The answer to decompose
            
        Returns:
            List of atomic statements
        """
        # Create new conversation to avoid state pollution
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        prompt = DECOMPOSITION_PROMPT.format(
            question=question,
            answer=answer
        )
        
        try:
            result = await conv.asend(prompt, returns=DecomposedStatements)
            return result.statements if result and result.statements else []
        except Exception as e:
            print(f"[ERROR] Failed to decompose statements: {e}")
            return []
    
    async def _classify_statements(
        self,
        question: str,
        answer_statements: List[str],
        ground_truth_statements: List[str]
    ) -> CorrectnessClassification:
        """
        Classify statements as TP/FP/FN
        
        Args:
            question: The original question
            answer_statements: Statements from the answer
            ground_truth_statements: Statements from ground truth
            
        Returns:
            CorrectnessClassification with TP/FP/FN lists
        """
        # Create new conversation to avoid state pollution
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)
        
        # Format statements for prompt
        answer_stmts_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(answer_statements)])
        gt_stmts_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(ground_truth_statements)])
        
        prompt = CLASSIFICATION_PROMPT.format(
            question=question,
            answer_statements=answer_stmts_text,
            ground_truth_statements=gt_stmts_text
        )
        
        try:
            result = await conv.asend(prompt, returns=CorrectnessClassification)
            return result
        except Exception as e:
            print(f"[ERROR] Failed to classify statements: {e}")
            return None
