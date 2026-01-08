"""
General-purpose filter using LLM-as-Judge

Strict quality control: better reject good cases than accept bad ones.
"""

import asyncio
from enum import Enum
import chak
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ...schemas.eval import EvalCase, EvalDataset
from .base import Filter, FilterReport, ValidationResult


class StrictnessLevel(str, Enum):
    """Context completeness evaluation strictness level"""
    STRICT = "strict"      # Must have direct textual support
    MODERATE = "moderate"  # Allow logical inference (e.g., contrapositive)
    LENIENT = "lenient"    # Allow reasonable domain inference


class _LLMDecision(BaseModel):
    """Internal: LLM judgment output format"""
    decision: str = Field(description="ACCEPT or REJECT")
    reason: str = Field(description="Brief explanation")
    failed_criteria: list[str] | None = Field(default=None, description="List of failed criterion names")


class GeneralFilter(Filter):
    """
    General-purpose filter using LLM-as-Judge
    
    Design Philosophy:
    - Better to reject good cases than accept bad ones (宁可错杀，不可放过)
    - No auto-repair, only accept/reject decisions
    - Use larger model for accurate judgment
    
    Features:
    - Concurrent execution with rate limiting
    - Automatic retry on transient errors
    - Error isolation: one case failure won't affect others
    - Progress tracking
    """
    
    def __init__(
        self,
        uri: str = "bailian/qwen-plus",
        api_key: str | None = None,
        concurrency: int = 3,
        strictness: StrictnessLevel = StrictnessLevel.MODERATE,
    ):
        """
        Initialize filter
        
        Args:
            uri: LLM URI for judgment (e.g., "bailian/qwen-plus")
            api_key: API key
            concurrency: Max concurrent LLM calls
            strictness: Context completeness evaluation strictness
                - STRICT: Answer must have direct textual support in contexts
                - MODERATE: Allow logical inference (recommended for multi-hop)
                - LENIENT: Allow reasonable domain knowledge inference
        """
        self.uri = uri
        self.api_key = api_key
        self.concurrency = concurrency
        self.strictness = strictness
    
    async def filter(self, dataset: EvalDataset) -> EvalDataset:
        """
        Apply strict filtering to dataset (async)
        
        Input: dataset (raw)
        Output: dataset (filtered)
        """
        rprint("\n[cyan]=== Starting Strict Dataset Filtering ===[/cyan]")
        rprint(f"[cyan]Judge Model: {self.uri}[/cyan]")
        rprint(f"[cyan]Total Cases: {len(dataset.cases)}[/cyan]\n")
        
        # Stage 1: LLM-based strict validation
        rprint("[bold]Stage 1:[/bold] LLM-as-Judge validation...")
        validation_results = await self._validate_cases(dataset.cases)
        
        # Separate approved and rejected
        approved_cases = [r.case for r in validation_results if r.decision.upper() == "ACCEPT"]
        rejected_cases = [r for r in validation_results if r.decision.upper() == "REJECT"]
        
        rprint(f"[green]✓[/green] Validated: {len(approved_cases)} accepted, {len(rejected_cases)} rejected\n")
        
        # Generate report
        report = FilterReport(
            total_cases=len(dataset.cases),
            accepted_cases=len(approved_cases),
            rejected_cases=len(rejected_cases),
            rejection_reasons=self._count_rejection_reasons(rejected_cases)
        )
        report.print_summary()
        
        # Print rejected cases details
        if rejected_cases:
            self._print_rejection_details(rejected_cases)
        
        # Return filtered dataset
        return EvalDataset(
            cases=approved_cases,
            domain=dataset.domain,
            generated_at=dataset.generated_at
        )
    
    async def _validate_cases(
        self,
        cases: list[EvalCase]
    ) -> list[ValidationResult]:
        """
        Validate cases using LLM-as-Judge with strict criteria
        
        Returns:
            List of validation results
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        conv = chak.Conversation(self.uri, api_key=self.api_key)
        
        async def judge_one_case_with_retry(
            case: EvalCase,
            index: int,
            progress: Progress,
            task_id
        ) -> ValidationResult:
            """Judge a single case with retry logic"""
            MAX_RETRIES = 3
            
            for attempt in range(MAX_RETRIES):
                async with semaphore:
                    try:
                        # Build prompt
                        prompt = self._build_judge_prompt(case)
                        
                        # Get LLM judgment
                        llm_decision = await conv.asend(prompt, returns=_LLMDecision)
                        
                        # Update progress
                        progress.update(task_id, advance=1)
                        
                        # Convert to ValidationResult
                        return ValidationResult(
                            decision=llm_decision.decision,
                            case=case,
                            reason=llm_decision.reason,
                            failed_criteria=llm_decision.failed_criteria or []
                        )
                            
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            # Retry on transient errors
                            await asyncio.sleep(1 * (attempt + 1))
                            continue
                        else:
                            # Final attempt failed, reject to be safe
                            progress.update(task_id, advance=1)
                            return ValidationResult(
                                decision="REJECT",
                                case=case,
                                reason=f"Validation error after {MAX_RETRIES} retries: {e}",
                                failed_criteria=["validation_error"]
                            )
            
            # Should never reach here
            progress.update(task_id, advance=1)
            return ValidationResult(
                decision="REJECT",
                case=case,
                reason="Unknown error",
                failed_criteria=["unknown_error"]
            )
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total} cases"),
        ) as progress:
            task_id = progress.add_task("Validating cases...", total=len(cases))
            
            # Judge all cases concurrently
            tasks = [
                judge_one_case_with_retry(case, i, progress, task_id)
                for i, case in enumerate(cases)
            ]
            results = await asyncio.gather(*tasks)
        
        return results
    
    def _build_judge_prompt(self, case: EvalCase) -> str:
        """Build strict judgment prompt for LLM-as-Judge"""
        # Format contexts
        contexts_str = ""
        for i, ctx in enumerate(case.ground_truth_contexts, 1):
            contexts_str += f"Context {i}:\n{ctx}\n\n"
        
        # Format source units
        source_units_str = ""
        for unit in case.source_units:
            source_units_str += f"- Unit {unit['unit_id']}:\n"
            source_units_str += f"  Summary: {unit.get('summary', 'N/A')}\n"
            source_units_str += f"  Keyphrases: {', '.join(unit.get('keyphrases', []))}\n\n"
        
        # Determine hop type
        hop_type = "multi-hop" if len(case.ground_truth_contexts) > 1 else "single-hop"
        
        prompt = f"""You are a STRICT quality inspector for RAG evaluation datasets.

Your task: Judge if this test case is **ACCEPTABLE** or should be **REJECTED**.

**CRITICAL PRINCIPLE**: Be VERY STRICT. When in doubt, REJECT.
Better to reject 10 good cases than accept 1 bad case.

---

## Test Case ({hop_type}):

**Question:**
{case.question}

**Ground Truth Answer:**
{case.ground_truth_answer}

**Ground Truth Contexts:**
{contexts_str}
**Source Units (metadata):**
{source_units_str}
---

## Evaluation Criteria (ALL must pass):

### 1. Context Completeness (最关键)
{self._get_context_completeness_criteria()}

### 2. Multi-hop Requirement (for {hop_type} cases)
{"Does the question require reasoning across MULTIPLE contexts?" if hop_type == "multi-hop" else "Can the question be answered from a SINGLE context?"}
- **REJECT if**: {"Question can be answered from single context alone" if hop_type == "multi-hop" else "Question requires multiple contexts (should be single-hop)"}

### 3. Context-Source Alignment
Do ground_truth_contexts match the content described in source_units?
- Check: Contexts should align with source units' summaries/keyphrases
- **REJECT if**: Obvious mismatch between contexts and source metadata

### 4. Question Quality
Is the question clear, realistic, and well-formed?
- **REJECT if**: Question is ambiguous or unclear
- **REJECT if**: Question contains artifacts like "As a [role], ..." (too artificial)
- **REJECT if**: Question is poorly written or confusing

---

## Output Format (JSON):

{{
    "decision": "ACCEPT" or "REJECT",
    "reason": "Brief explanation (1 sentence)",
    "failed_criteria": ["criterion_name"] or null
}}

**Examples:**
- ACCEPT: {{
    "decision": "ACCEPT",
    "reason": "All criteria passed, contexts fully support answer",
    "failed_criteria": null
}}

- REJECT: {{
    "decision": "REJECT",
    "reason": "Answer mentions FHA loans but contexts discuss conventional mortgages",
    "failed_criteria": ["context_completeness"]
}}

**Remember**: When uncertain, choose REJECT. Quality over quantity.
"""
        
        return prompt
    
    def _get_context_completeness_criteria(self) -> str:
        """Get context completeness criteria based on strictness level"""
        if self.strictness == StrictnessLevel.STRICT:
            return """Do the contexts FULLY support the answer?
- Check: Every fact in the answer must be traceable to contexts
- **REJECT if**: Any answer fact cannot be found in contexts
- **REJECT if**: Answer mentions concepts not present in contexts
- **No inference allowed**: Answer must have direct textual support"""
        
        elif self.strictness == StrictnessLevel.MODERATE:
            return """Do the contexts support the answer through direct statements or logical inference?
- Check: Answer facts should be directly stated OR logically inferable from contexts
- **ACCEPT**: Logical inferences like contrapositive ("low score → high payment" implies "high score → low payment")
- **REJECT if**: Answer requires domain knowledge not present in contexts
- **REJECT if**: Answer introduces entirely new concepts
- **Recommended for multi-hop evaluation**: Allows reasonable information synthesis"""
        
        else:  # LENIENT
            return """Do the contexts reasonably support the answer?
- Check: Answer should be consistent with context information
- **ACCEPT**: Reasonable domain inferences based on context
- **ACCEPT**: Combining information across contexts with common knowledge
- **REJECT if**: Answer directly contradicts contexts
- **REJECT if**: Answer is completely unsupported
- **Use with caution**: May allow overly creative answers"""
    
    def _count_rejection_reasons(self, rejected_cases: list[ValidationResult]) -> dict[str, int]:
        """Count rejection reasons from rejected cases"""
        reason_counts = {}
        for result in rejected_cases:
            for criterion in result.failed_criteria:
                reason_counts[criterion] = reason_counts.get(criterion, 0) + 1
        
        return reason_counts
    
    def _print_rejection_details(self, rejected_cases: list[ValidationResult]):
        """Print detailed information about rejected cases"""
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        rprint("\n[bold red]Rejected Cases Details:[/bold red]\n")
        
        for i, result in enumerate(rejected_cases, 1):
            case = result.case
            
            # Format contexts
            contexts_display = ""
            for ctx_idx, ctx in enumerate(case.ground_truth_contexts, 1):
                ctx_preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
                contexts_display += f"\n[dim]Context {ctx_idx}:[/dim]\n{ctx_preview}\n"
            
            if not contexts_display:
                contexts_display = "[dim]No contexts available[/dim]"
            
            content = f"""[yellow]Question:[/yellow]
{case.question}

[green]Ground Truth Answer:[/green]
{case.ground_truth_answer}

[cyan]Ground Truth Contexts:[/cyan]{contexts_display}
[red]Rejection Reason:[/red]
{result.reason}

[dim]Failed Criteria:[/dim] {', '.join(result.failed_criteria)}"""
            
            console.print(Panel(
                content,
                title=f"[bold white]Rejected Case #{i}[/bold white]",
                border_style="red",
                padding=(1, 2)
            ))
            console.print()
