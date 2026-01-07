"""Evaluation data schemas"""

from pydantic import BaseModel, Field
from typing import Any
import json
import csv
from pathlib import Path


class EvalCase(BaseModel):
    """
    Single evaluation case
    
    Self-contained design: includes all information needed for evaluation
    and traceability without external lookups.
    
    Attributes:
        question: Generated question
        ground_truth_answer: Expected answer (ground truth, for comparison)
        ground_truth_contexts: Reference contexts for ground truth answer
        answer: AI's answer (filled during evaluation, initially None)
        retrieved_contexts: Retrieved contexts (for RAG evaluation, filled during eval)
        source_units: List of source units information (supports multi-hop)
        persona: Persona used for generation
        generation_params: Generation parameters for analysis
    """
    question: str = Field(description="Generated question")
    ground_truth_answer: str = Field(description="Expected answer (ground truth)")
    ground_truth_contexts: list[str] = Field(
        description="Reference contexts for ground truth answer"
    )
    
    # For evaluation (filled during eval)
    answer: str | None = Field(
        default=None,
        description="AI's generated answer (filled during evaluation)"
    )
    retrieved_contexts: list[str] | None = Field(
        default=None,
        description="Retrieved contexts by RAG system (filled during evaluation)"
    )
    
    # Traceability: store complete data, not just IDs
    source_units: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source units information (supports single-hop and multi-hop)"
    )
    
    persona: dict[str, Any] | None = Field(
        default=None,
        description="Persona information used for generation"
    )
    
    generation_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Generation parameters (style, length, etc.)"
    )


class EvalDataset(BaseModel):
    """
    Evaluation dataset
    
    Collection of evaluation cases with export capabilities.
    """
    cases: list[EvalCase] = Field(description="Evaluation cases")
    domain: str | None = Field(default=None, description="Domain")
    generated_at: str | None = Field(default=None, description="Generation timestamp")
    
    def __len__(self):
        return len(self.cases)
    
    def to_json(self, path: str):
        """Export as JSON file"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                self.model_dump(),
                f,
                ensure_ascii=False,
                indent=2
            )
    
    def to_jsonl(self, path: str):
        """Export as JSONL file (one case per line)"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for case in self.cases:
                f.write(json.dumps(case.model_dump(), ensure_ascii=False) + '\n')
    
    def to_csv(self, path: str):
        """Export as CSV file"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            if not self.cases:
                return
            
            # Basic fields for CSV (complex nested data goes to JSON columns)
            fieldnames = [
                'question',
                'ground_truth_answer',
                'answer',  # AI's answer (may be empty)
                'ground_truth_contexts',  # JSON string
                'retrieved_contexts',  # JSON string
                'source_unit_ids',  # comma-separated
                'persona_name',
                'generation_params'  # JSON string
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for case in self.cases:
                # Extract source unit IDs
                unit_ids = [u.get('unit_id', '') for u in case.source_units]
                
                row = {
                    'question': case.question,
                    'ground_truth_answer': case.ground_truth_answer,
                    'answer': case.answer or '',  # May be empty
                    'ground_truth_contexts': json.dumps(case.ground_truth_contexts, ensure_ascii=False),
                    'retrieved_contexts': json.dumps(case.retrieved_contexts or [], ensure_ascii=False),
                    'source_unit_ids': ','.join(unit_ids),
                    'persona_name': case.persona.get('name') if case.persona else '',
                    'generation_params': json.dumps(case.generation_params, ensure_ascii=False)
                }
                writer.writerow(row)
