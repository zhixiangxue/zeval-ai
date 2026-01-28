"""
Splitter examples - Demonstrate MarkdownHeaderSplitter and ChunkSplitter

This example shows:
1. MarkdownHeaderSplitter - Split markdown by headers with context path
2. ChunkSplitter - Simple token-based splitting using Chonkie
3. Pipeline usage - Combine splitters for optimal results
"""

from zeval.schemas.markdown import Markdown
from zeval.synthetic_data.splitters import MarkdownHeaderSplitter, ChunkSplitter


def markdown_header_splitter_example():
    """Example: Split markdown by headers"""
    print("\n" + "="*60)
    print("Example 1: MarkdownHeaderSplitter")
    print("="*60)
    
    # Sample markdown content with nested headers
    markdown_content = """# Introduction

This is the introduction section with some overview text.

## Background

Here we discuss the background and motivation for this project.

### Technical Details

Some technical details about the implementation.

## Methodology

Our approach involves several key steps:

1. Data collection
2. Data processing
3. Analysis

### Data Collection

Details about how we collected the data.

### Data Processing

Information about our data processing pipeline.

# Results

The results section contains our findings.

## Performance Metrics

Here are the key performance metrics we measured.
"""
    
    # Create markdown document
    doc = Markdown(content=markdown_content)
    
    # Create splitter
    splitter = MarkdownHeaderSplitter(
        header_path_separator="/",
        include_header_in_content=True
    )
    
    # Split document
    units = doc.split(splitter)
    
    print(f"\n✓ Split into {len(units)} units\n")
    
    # Display results
    for i, unit in enumerate(units, 1):
        context_path = unit.metadata.context_path if unit.metadata else "No context"
        preview = unit.content[:80].replace("\n", " ")
        print(f"Unit {i}:")
        print(f"  Context Path: {context_path}")
        print(f"  Content: {preview}...")
        print(f"  Unit ID: {unit.unit_id}")
        print()


def chunk_splitter_example():
    """Example: Simple token-based splitting with Chonkie"""
    print("\n" + "="*60)
    print("Example 2: ChunkSplitter (Token-based)")
    print("="*60)
    
    # Long text content
    long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
the natural intelligence displayed by humans and animals. Leading AI textbooks define 
the field as the study of "intelligent agents": any device that perceives its environment 
and takes actions that maximize its chance of successfully achieving its goals.

The term "artificial intelligence" is often used to describe machines (or computers) that 
mimic "cognitive" functions that humans associate with the human mind, such as "learning" 
and "problem solving". As machines become increasingly capable, tasks considered to require 
"intelligence" are often removed from the definition of AI, a phenomenon known as the AI 
effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet."

Modern machine learning techniques are highly effective in a wide variety of tasks and 
have been successfully applied in many industries. Deep learning has been particularly 
effective in image recognition, natural language processing, and game playing. These 
advances have led to a renaissance in AI research and funding.
""" * 5  # Repeat to make it longer
    
    # Create markdown document
    doc = Markdown(content=long_text)
    
    # Create splitter with small chunks for demo
    splitter = ChunkSplitter(
        chunk_size=200,      # 200 tokens per chunk
        chunk_overlap=20     # 20 tokens overlap
    )
    
    # Split document
    units = doc.split(splitter)
    
    print(f"\n✓ Split into {len(units)} chunks\n")
    
    # Display first 3 chunks
    for i, unit in enumerate(units[:3], 1):
        preview = unit.content[:100].replace("\n", " ").strip()
        metadata = unit.metadata.custom if unit.metadata and unit.metadata.custom else {}
        actual_tokens = metadata.get('actual_tokens', 'N/A')
        chunk_index = metadata.get('chunk_index', 'N/A')
        
        print(f"Chunk {i}:")
        print(f"  Index: {chunk_index}")
        print(f"  Actual Tokens: {actual_tokens}")
        print(f"  Content: {preview}...")
        print()
    
    if len(units) > 3:
        print(f"... and {len(units) - 3} more chunks")


def pipeline_example():
    """Example: Combine splitters in pipeline"""
    print("\n" + "="*60)
    print("Example 3: Pipeline - Header Split + Token Split")
    print("="*60)
    
    # Markdown with large sections
    markdown_content = """# Chapter 1: Introduction

""" + "This is a very long introduction section. " * 100 + """

## Section 1.1: Background

""" + "Background information with lots of details. " * 100 + """

# Chapter 2: Methods

""" + "Detailed methodology description. " * 100
    
    # Create document
    doc = Markdown(content=markdown_content)
    
    # Step 1: Split by headers
    print("\nStep 1: Split by headers...")
    header_splitter = MarkdownHeaderSplitter()
    header_units = doc.split(header_splitter)
    print(f"  Result: {len(header_units)} header-based units")
    
    # Step 2: Further split large units by tokens
    print("\nStep 2: Split large units by tokens...")
    chunk_splitter = ChunkSplitter(chunk_size=300, chunk_overlap=30)
    final_units = chunk_splitter.split(header_units)
    print(f"  Result: {len(final_units)} final units")
    
    print("\n✓ Pipeline complete!\n")
    
    # Show first few results
    for i, unit in enumerate(final_units[:3], 1):
        context = unit.metadata.context_path if unit.metadata else "No context"
        chunk_info = ""
        if unit.metadata and unit.metadata.custom:
            chunk_idx = unit.metadata.custom.get('chunk_index')
            if chunk_idx is not None:
                chunk_info = f" [Chunk {chunk_idx}]"
        
        preview = unit.content[:80].replace("\n", " ").strip()
        print(f"Unit {i}:")
        print(f"  Context: {context}{chunk_info}")
        print(f"  Content: {preview}...")
        print()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("SPLITTER EXAMPLES")
    print("="*60)
    print("\nDemonstrating different splitting strategies:")
    print("1. MarkdownHeaderSplitter - Semantic splitting by headers")
    print("2. ChunkSplitter - Simple token-based splitting")
    print("3. Pipeline - Combine both for optimal results")
    
    try:
        # Run examples
        markdown_header_splitter_example()
        chunk_splitter_example()
        pipeline_example()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED!")
        print("="*60 + "\n")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install chonkie tiktoken")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
