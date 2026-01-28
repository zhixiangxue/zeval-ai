"""
DoclingReader Example

Demonstrates basic usage of DoclingReader for PDF parsing.
Shows different pipeline configurations and their outputs.
"""

from pathlib import Path
from zeval.synthetic_data.readers.docling import DoclingReader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions


def basic_usage():
    """Example 1: Basic usage with default settings"""
    print("\n=== Example 1: Basic Usage ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Configure PDF pipeline with auto-detect accelerator
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA  # Auto-detect CUDA or fallback to CPU
    )
    
    # Create reader with pipeline options
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    
    # Read PDF file
    doc = reader.read(pdf_path)
    
    # Access document content
    print(f"Content length: {len(doc.content)} characters")
    print(f"Number of pages: {len(doc.pages)}")
    print(f"Page count: {doc.metadata.custom.get('page_count')}")
    
    # Access page content
    for page in doc.pages[:3]:  # Show first 3 pages
        print(f"\n--- Page {page.page_number} ---")
        print(f"Content preview: {page.content[:200]}...")
    
    return doc


def with_pipeline_options():
    """Example 2: Custom pipeline options"""
    print("\n=== Example 2: Custom Pipeline Options ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Configure PDF pipeline
    pdf_options = PdfPipelineOptions(
        do_ocr=True,  # Enable OCR for scanned PDFs
        do_table_structure=True  # Enable table structure recognition
    )
    
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(pdf_path)
    
    print(f"Parsed with OCR: {len(doc.content)} characters")
    print(f"Pages: {len(doc.pages)}")
    
    return doc


def with_gpu_acceleration():
    """Example 3: GPU acceleration"""
    print("\n=== Example 3: GPU Acceleration ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Configure GPU acceleration
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.AUTO  # or CUDA, CPU, MPS
    )
    
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    doc = reader.read(pdf_path)
    
    print(f"Parsed with GPU: {len(doc.content)} characters")
    
    return doc


def save_output(doc, output_path: str = "output.md"):
    """Example 4: Save parsed content"""
    print(f"\n=== Example 4: Save Output ===")
    
    # Save full document markdown
    output_file = Path(output_path)
    output_file.write_text(doc.content, encoding="utf-8")
    print(f"Saved to: {output_file.absolute()}")
    
    # Save individual pages
    pages_dir = output_file.parent / "pages"
    pages_dir.mkdir(exist_ok=True)
    
    for page in doc.pages:
        page_file = pages_dir / f"page_{page.page_number}.md"
        page_file.write_text(page.content, encoding="utf-8")
    
    print(f"Saved {len(doc.pages)} pages to: {pages_dir.absolute()}")


def inspect_metadata(doc):
    """Example 5: Inspect document metadata"""
    print("\n=== Example 5: Document Metadata ===")
    
    print(f"Source: {doc.metadata.source}")
    print(f"File type: {doc.metadata.file_type}")
    print(f"File name: {doc.metadata.file_name}")
    print(f"File size: {doc.metadata.file_size} bytes")
    print(f"Content length: {doc.metadata.content_length} characters")
    print(f"Reader: {doc.metadata.reader_name}")
    
    print("\nCustom metadata:")
    for key, value in doc.metadata.custom.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=" * 60)
    print("DoclingReader Examples")
    print("=" * 60)
    
    # Run examples (uncomment the ones you want to try)
    doc = basic_usage()
    # doc = with_pipeline_options()
    # doc = with_gpu_acceleration()
    # save_output(doc)
    # inspect_metadata(doc)
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
