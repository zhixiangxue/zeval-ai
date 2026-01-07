"""
Test TransformPipeline with extractors
"""

import asyncio
import os
from dotenv import load_dotenv
from zeval.schemas.markdown import Markdown
from zeval.schemas.base import DocumentMetadata, BaseUnit, UnitMetadata
from zeval.synthetic_data.transforms import TransformPipeline
from zeval.synthetic_data.transforms.extractors import (
    SummaryExtractor,
    KeyphrasesExtractor,
    EntitiesExtractor
)

# Load environment variables
load_dotenv()


def create_test_units():
    """Create test units for transformation"""
    
    # Create test markdown document
    markdown_content = """# Artificial Intelligence

Artificial intelligence (AI) is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately.

## Machine Learning

Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.

## Deep Learning

Deep learning uses neural networks with multiple layers to process complex patterns in data. It has revolutionized computer vision, natural language processing, and speech recognition.
"""
    
    doc = Markdown(
        content=markdown_content,
        metadata=DocumentMetadata(
            source="test_ai.md",
            source_type="local",
            file_type="markdown",
            content_length=len(markdown_content)
        )
    )
    
    # Manually create units (simulating splitter output)
    units = [
        BaseUnit(
            unit_id="unit_1",
            content="# Artificial Intelligence\n\nArtificial intelligence (AI) is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately.",
            source_doc_id=doc.doc_id,
            metadata=UnitMetadata(context_path="Artificial Intelligence")
        ),
        BaseUnit(
            unit_id="unit_2",
            content="## Machine Learning\n\nMachine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.",
            source_doc_id=doc.doc_id,
            metadata=UnitMetadata(context_path="Artificial Intelligence/Machine Learning")
        ),
        BaseUnit(
            unit_id="unit_3",
            content="## Deep Learning\n\nDeep learning uses neural networks with multiple layers to process complex patterns in data. It has revolutionized computer vision, natural language processing, and speech recognition.",
            source_doc_id=doc.doc_id,
            metadata=UnitMetadata(context_path="Artificial Intelligence/Deep Learning")
        )
    ]
    
    return units


async def main():
    """Test transformation pipeline"""
    
    # Check for API key
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("Error: BAILIAN_API_KEY environment variable not set")
        print("Please set it with: export OPENAIBAILIAN_API_KEY_API_KEY='your-key'")
        return
    
    print("\n" + "="*60)
    print("Testing TransformPipeline")
    print("="*60)
    
    # 1. Create test units
    units = create_test_units()
    print(f"\n✓ Created {len(units)} test units")
    
    # 2. Create extractors
    extractors = [
        SummaryExtractor(
            model_uri="bailian/qwen-plus",
            api_key=api_key,
            max_sentences=1
        ),
        KeyphrasesExtractor(
            model_uri="bailian/qwen-plus",
            api_key=api_key,
            max_num=3
        ),
        EntitiesExtractor(
            model_uri="bailian/qwen-plus",
            api_key=api_key,
            max_num=5
        )
    ]
    print(f"✓ Created {len(extractors)} extractors")
    
    # 3. Run transformation pipeline
    print("\n⏳ Running transformation pipeline...")
    pipeline = TransformPipeline(
        extractors=extractors,
        max_concurrency=5  # Lower concurrency for testing
    )
    
    enriched_units = await pipeline.transform(units)
    
    print(f"✓ Transformation completed")
    
    # 4. Display results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    for i, unit in enumerate(enriched_units, 1):
        print(f"\n--- Unit {i}: {unit.metadata.context_path} ---")
        print(f"Content (first 100 chars): {unit.content[:100]}...")
        
        print("\nExtracted properties:")
        
        if unit.summary:
            print(f"  Summary: {unit.summary}")
        
        if unit.keyphrases:
            print(f"  Keyphrases: {unit.keyphrases}")
        
        if unit.entities:
            print(f"  Entities: {unit.entities}")
        
        if not unit.summary and not unit.keyphrases and not unit.entities:
            print("  (No extracted properties)")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
