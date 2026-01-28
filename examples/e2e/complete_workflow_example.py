"""Complete E2E Test: PDF -> Read -> Split -> Transform -> Generate -> Evaluate -> Report

This script demonstrates the complete workflow:
1. Read a PDF file using DoclingReader
2. Split into units using MarkdownHeaderSplitter
3. Enrich units with extractors (summary, keyphrases, entities)
4. Generate personas
5. Generate single-hop Q&A pairs
6. Evaluate with all metrics
7. Generate analysis report

Prerequisites:
    - OPENAI_API_KEY environment variable must be set
    - A sample PDF file (will be downloaded if not exists)

Usage:
    python examples/test_e2e_complete.py
    
    # Or specify your own PDF:
    python examples/test_e2e_complete.py --pdf /path/to/your/document.pdf
"""

import asyncio
import os
import sys
import ssl
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pydantic import BaseModel, Field
from chak import Conversation

# Disable SSL verification for HuggingFace downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress HTTP request logs that break progress bar rendering
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Import all required components
from zeval.synthetic_data.readers.docling import DoclingReader
from zeval.synthetic_data.splitters import MarkdownHeaderSplitter
from zeval.synthetic_data.transforms.extractors import (
    SummaryExtractor,
    KeyphrasesExtractor,
    EntitiesExtractor
)
from zeval.synthetic_data.generators.persona import generate_personas, Persona
from zeval.synthetic_data.generators.single_hop import generate_single_hop
from zeval.evaluation.metrics import (
    Faithfulness,
    ContextRelevance,
    ContextRecall,
    ContextPrecision,
    AnswerRelevancy,
    AnswerCorrectness,
)
from zeval.evaluation.runner import MetricRunner
from zeval.evaluation.reporter import EvaluationReporter
from zeval.schemas.eval import EvalDataset, EvalCase
from zeval.schemas.unit import BaseUnit

console = Console()


# ============================================================================
# Mock RAG Server Implementation
# ============================================================================

class RAGResponse(BaseModel):
    """Structured response from RAG system"""
    answer: str = Field(description="The generated answer to the question")
    retrieved_contexts: list[str] = Field(
        description="List of 2-4 most relevant context passages retrieved from the knowledge base"
    )


async def call_mock_rag(
    question: str,
    llm_uri: str,
    api_key: str
) -> RAGResponse:
    """
    Mock RAG server that uses LLM to simulate RAG system behavior
    
    Args:
        question: User's question
        llm_uri: LLM endpoint
        api_key: API key
        
    Returns:
        RAGResponse with answer and retrieved_contexts
    """
    # System message - define RAG's role and expertise
    system_message = """You are an expert RAG (Retrieval-Augmented Generation) system specialized in US residential mortgage lending.

Your expertise covers:
- FHA, VA, USDA, and Conventional loan programs
- Underwriting guidelines from Fannie Mae, Freddie Mac, and FHA
- Down payment requirements and PMI regulations
- Credit score impacts on mortgage eligibility
- Debt-to-Income (DTI) ratio calculations
- Mortgage insurance and closing costs

Your task is to provide accurate, professional answers and simulate retrieved context passages that would support your answer."""
    
    # User prompt
    user_prompt = f"""User Question: {question}

Please provide:
1. A professional answer based on your expertise in US residential mortgage lending
2. 2-4 realistic context passages that would typically be retrieved from a mortgage lending knowledge base to support this answer"""
    
    # Create Conversation with system message
    conv = Conversation(
        model_uri=llm_uri, 
        api_key=api_key,
        system_message=system_message
    )
    
    try:
        response = await asyncio.wait_for(
            conv.asend(user_prompt, returns=RAGResponse),
            timeout=30.0
        )
        return response
    except Exception as e:
        # Fallback: return a simple response
        print(f"[WARNING] RAG call failed: {e}, using fallback")
        return RAGResponse(
            answer="I don't have enough information to answer this question.",
            retrieved_contexts=["No context available"]
        )


async def fill_dataset_with_rag(
    dataset: EvalDataset,
    llm_uri: str,
    api_key: str,
    max_concurrent: int = 5
) -> None:
    """
    Fill dataset cases with RAG-generated answers and retrieved contexts
    
    Args:
        dataset: Dataset with generated questions
        llm_uri: LLM endpoint
        api_key: API key
        max_concurrent: Maximum concurrent RAG calls (default: 5)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task_id = progress.add_task("Processing RAG calls", total=len(dataset.cases))
        
        async def process_case(case: EvalCase):
            async with semaphore:
                rag_response = await call_mock_rag(
                    question=case.question,
                    llm_uri=llm_uri,
                    api_key=api_key
                )
                case.answer = rag_response.answer
                case.retrieved_contexts = rag_response.retrieved_contexts
                progress.update(task_id, advance=1)
        
        # Execute all tasks concurrently
        tasks = [process_case(case) for case in dataset.cases]
        await asyncio.gather(*tasks)


# ============================================================================
# Domain-Specific Persona
# ============================================================================

class HomeBuyerPersona(Persona):
    """US Home Buyer persona with financial attributes"""
    credit_score: int = Field(
        description="Credit score (300-850), affects mortgage eligibility and interest rates"
    )
    dti_ratio: float = Field(
        description="Debt-to-Income ratio as percentage (typical max is 43%)"
    )
    down_payment_percent: float = Field(
        description="Down payment as percentage of home price (typically 3-20%)"
    )
    budget_range: str = Field(
        description="Home price budget range (e.g., '$300K-$500K')"
    )


# ============================================================================


async def main():
    """Main E2E workflow"""
    
    # ============================================================================
    # Setup and Configuration
    # ============================================================================
    console.print("\n" + "="*80)
    console.print("[bold cyan]Complete E2E Test: PDF to Evaluation Report[/bold cyan]")
    console.print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]✗ Error: OPENAI_API_KEY not found in environment[/red]")
        console.print("  Please set it in your .env file or environment")
        return
    
    llm_uri = "openai/gpt-4o-mini"
    
    # ============================================================================
    # Interactive PDF Input
    # ============================================================================
    console.print("[bold]请输入PDF文件路径:[/bold]")
    pdf_input = input("> ").strip()
    
    if not pdf_input:
        console.print("[red]✗ Error: PDF path cannot be empty[/red]")
        return
    
    # Clean path: remove PowerShell artifacts and quotes
    # PowerShell may add "& '" prefix for paths with spaces
    if pdf_input.startswith("& '") or pdf_input.startswith('& "'):
        pdf_input = pdf_input[3:]  # Remove "& '" or '& "'
    
    # Remove quotes (from drag-and-drop or PowerShell)
    pdf_input = pdf_input.strip('"').strip("'")
    
    pdf_path = Path(pdf_input)
    if not pdf_path.exists():
        console.print(f"[red]✗ Error: PDF file not found: {pdf_path}[/red]")
        return
    
    # Setup output paths
    workspace = Path.cwd()
    output_base_dir = workspace / "output"
    output_base_dir.mkdir(exist_ok=True)
    
    # Create timestamped report directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = output_base_dir / f"report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        f"[bold]Configuration[/bold]\n\n"
        f"PDF File: {pdf_path}\n"
        f"LLM: {llm_uri}\n"
        f"Report Output: {report_dir}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title="⚙️ Setup",
        border_style="blue"
    ))
    
    # ============================================================================
    # Stage 1: Read PDF
    # ============================================================================
    console.print("\n[bold yellow]Stage 1: Reading PDF[/bold yellow]")
    console.print("-" * 80)
    
    console.print("📖 Loading PDF with DoclingReader...")
    
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    
    # Use CPU for compatibility
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CPU
    )
    
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    document = reader.read(str(pdf_path))
    
    console.print(f"[green]✓[/green] Document loaded")
    console.print(f"  - Content length: {len(document.content):,} characters")
    console.print(f"  - Pages: {len(document.pages)}")
    console.print(f"  - File: {document.metadata.file_name}")
    
    # Save intermediate result
    md_output = report_dir / "01_document_content.md"
    with md_output.open("w", encoding="utf-8") as f:
        f.write(document.content)
    console.print(f"  - Saved to: {md_output}")
    
    # ============================================================================
    # Stage 2: Split into Units
    # ============================================================================
    console.print("\n[bold yellow]Stage 2: Splitting Document[/bold yellow]")
    console.print("-" * 80)
    
    console.print("✂️  Splitting by markdown headers...")
    
    splitter = MarkdownHeaderSplitter()
    units = document.split(splitter)
    
    console.print(f"[green]✓[/green] Split into {len(units)} units")
    
    # Show sample units
    console.print("\n  Sample units:")
    for i, unit in enumerate(units[:3], 1):
        preview = unit.content[:100].replace("\n", " ")
        console.print(f"    {i}. [{unit.metadata.context_path}] {preview}...")
    
    if len(units) > 3:
        console.print(f"    ... and {len(units) - 3} more units")
    
    # ============================================================================
    # Stage 3: Transform (Enrich Units)
    # ============================================================================
    console.print("\n[bold yellow]Stage 3: Enriching Units[/bold yellow]")
    console.print("-" * 80)
    
    console.print("🔍 Extracting summaries, keyphrases, and entities...")
    
    extractor = (
        SummaryExtractor(model_uri=llm_uri, api_key=api_key, max_sentences=2)
        | KeyphrasesExtractor(model_uri=llm_uri, api_key=api_key, max_num=5)
        | EntitiesExtractor(model_uri=llm_uri, api_key=api_key, max_num=5)
    )
    
    enriched_units = await extractor.transform(units, max_concurrency=3)
    
    console.print(f"[green]✓[/green] Enriched {len(enriched_units)} units")
    
    # Show enrichment sample
    if enriched_units:
        sample = enriched_units[0]
        console.print("\n  Sample enriched unit:")
        console.print(f"    Path: {sample.metadata.context_path}")
        if sample.summary:
            console.print(f"    Summary: {sample.summary[:100]}...")
        if sample.keyphrases:
            console.print(f"    Keyphrases: {', '.join(sample.keyphrases[:3])}")
        if sample.entities:
            console.print(f"    Entities: {', '.join(sample.entities[:3])}")
    
    # ============================================================================
    # Stage 4: Generate Personas
    # ============================================================================
    console.print("\n[bold yellow]Stage 4: Generating Personas[/bold yellow]")
    console.print("-" * 80)
    
    console.print("👥 Creating diverse user personas...")
    
    # Use domain-specific persona for US home buying
    domain = "US residential real estate and home buying process"
    
    personas = await generate_personas(
        llm_uri=llm_uri,
        api_key=api_key,
        domain=domain,
        num_personas=3,
        persona_model=HomeBuyerPersona  # Use custom persona with financial fields
    )
    
    console.print(f"[green]✓[/green] Generated {len(personas)} personas")
    for p in personas:
        console.print(f"  • [cyan]{p.name}[/cyan] - {p.expertise_level}")
    
    # ============================================================================
    # Stage 5: Generate Q&A Dataset
    # ============================================================================
    console.print("\n[bold yellow]Stage 5: Generating Q&A Dataset[/bold yellow]")
    console.print("-" * 80)
    
    console.print("💭 Generating single-hop questions...")
    
    # Limit to first N units for faster testing
    max_units = min(10, len(enriched_units))
    units_for_generation = enriched_units[:max_units]
    
    console.print(f"  Using {len(units_for_generation)} units for generation")
    
    dataset = await generate_single_hop(
        llm_uri=llm_uri,
        api_key=api_key,
        units=units_for_generation,
        personas=personas,
        num_cases=15,  # Generate 15 Q&A pairs
        domain=domain
    )
    
    console.print(f"[green]✓[/green] Generated {len(dataset.cases)} evaluation cases")
    
    # Save dataset
    dataset_file = report_dir / "05_generated_dataset.json"
    dataset.to_json(str(dataset_file))
    console.print(f"  - Saved to: {dataset_file}")
    
    # Show sample questions
    console.print("\n  Sample questions:")
    for i, case in enumerate(dataset.cases[:3], 1):
        console.print(f"    {i}. {case.question[:80]}...")
    
    # ============================================================================
    # Stage 5.5: Call Mock RAG to Fill Answer and Retrieved Contexts
    # ============================================================================
    console.print("\n[bold yellow]Stage 5.5: Calling Mock RAG Server[/bold yellow]")
    console.print("-" * 80)
    
    console.print("🤖 Simulating RAG system calls...\n")
    
    await fill_dataset_with_rag(
        dataset=dataset,
        llm_uri=llm_uri,
        api_key=api_key,
        max_concurrent=5
    )
    
    console.print(f"\n[green]✓[/green] All {len(dataset.cases)} cases processed by RAG")
    
    # ============================================================================
    # Stage 6: Evaluate with All Metrics
    # ============================================================================
    console.print("\n[bold yellow]Stage 6: Running Evaluation[/bold yellow]")
    console.print("-" * 80)
    
    console.print("📊 Evaluating with all metrics...\n")
    
    metrics = [
        Faithfulness(llm_uri=llm_uri, api_key=api_key),
        ContextRelevance(llm_uri=llm_uri, api_key=api_key),
        ContextRecall(llm_uri=llm_uri, api_key=api_key),
        ContextPrecision(llm_uri=llm_uri, api_key=api_key),
        AnswerRelevancy(llm_uri=llm_uri, api_key=api_key),
        AnswerCorrectness(llm_uri=llm_uri, api_key=api_key),
    ]
    
    runner = MetricRunner(metrics=metrics)
    await runner.run(dataset)
    
    console.print(f"\n[green]✓[/green] Evaluation completed")
    
    # Show sample results
    if dataset.cases:
        sample_case = dataset.cases[0]
        console.print("\n  Sample evaluation result:")
        for metric_name, result in sample_case.results.items():
            console.print(f"    {metric_name}: {result.score:.2f}")
    
    # Save evaluated dataset
    evaluated_file = report_dir / "06_evaluated_dataset.json"
    dataset.to_json(str(evaluated_file))
    console.print(f"  - Saved to: {evaluated_file}")
    
    # ============================================================================
    # Stage 7: Generate Report
    # ============================================================================
    console.print("\n[bold yellow]Stage 7: Generating Report[/bold yellow]")
    console.print("-" * 80)
    
    console.print("📝 Creating analysis report...")
    
    reporter = EvaluationReporter(llm_uri=llm_uri, api_key=api_key)
    
    # Pass the report directory directly
    report_content = await reporter.generate_report(
        dataset_path=str(evaluated_file),
        output_path=str(report_dir)
    )
    
    console.print(f"[green]✓[/green] Report generated")
    console.print(f"  - All outputs saved to: {report_dir}/")
    console.print(f"  - Report size: {len(report_content):,} characters")
    
    # ============================================================================
    # Summary
    # ============================================================================
    console.print("\n" + "="*80)
    console.print("[bold green]✨ E2E Test Completed Successfully![/bold green]")
    console.print("="*80 + "\n")
    
    console.print(Panel.fit(
        f"[bold]Pipeline Summary[/bold]\n\n"
        f"✓ Document processed: {len(document.pages)} pages\n"
        f"✓ Units created: {len(units)}\n"
        f"✓ Units enriched: {len(enriched_units)}\n"
        f"✓ Personas generated: {len(personas)}\n"
        f"✓ Q&A pairs created: {len(dataset.cases)}\n"
        f"✓ RAG calls completed: {len(dataset.cases)}\n"
        f"✓ Metrics evaluated: {len(metrics)}\n"
        f"✓ Report generated\n\n"
        f"[cyan]All outputs in: {report_dir}/[/cyan]",
        title="📊 Results",
        border_style="green"
    ))
    
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"  Open the report folder: {report_dir}")
    console.print(f"  All outputs are in one place:")
    console.print(f"    - 01_document_content.md")
    console.print(f"    - 05_generated_dataset.json")
    console.print(f"    - 06_evaluated_dataset.json")
    console.print(f"    - evaluation_report.md")
    console.print(f"    - evaluation_report.xlsx")
    console.print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
