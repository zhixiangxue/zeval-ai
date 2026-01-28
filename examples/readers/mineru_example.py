"""
MinerUReader Example

Demonstrates usage of MinerUReader for high-accuracy PDF parsing.
MinerU offers multiple backends with different accuracy/speed tradeoffs.
"""

from pathlib import Path
from zeval.synthetic_data.readers.mineru import MinerUReader


def basic_usage():
    """Example 1: Basic usage with default hybrid backend"""
    print("\n=== Example 1: Basic Usage (Hybrid Backend) ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Default uses hybrid-auto-engine (requires GPU)
    reader = MinerUReader()
    doc = reader.read(pdf_path)
    
    print(f"Content length: {len(doc.content)} characters")
    print(f"Number of pages: {len(doc.pages)}")
    
    # Access page content
    for page in doc.pages[:3]:  # Show first 3 pages
        print(f"\n--- Page {page.page_number} ---")
        print(f"Content preview: {page.content[:200]}...")
    
    return doc


def cpu_only_pipeline():
    """Example 2: CPU-only pipeline backend"""
    print("\n=== Example 2: CPU-Only Pipeline ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Use pipeline backend for CPU-only environments
    reader = MinerUReader(backend="pipeline")
    doc = reader.read(pdf_path)
    
    print(f"Parsed with pipeline backend: {len(doc.content)} characters")
    print(f"Pages: {len(doc.pages)}")
    
    return doc


def vlm_backend():
    """Example 3: Pure VLM backend for highest accuracy"""
    print("\n=== Example 3: VLM Backend (Highest Accuracy) ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Use VLM for highest accuracy (requires GPU with 8GB+ VRAM)
    reader = MinerUReader(backend="vlm-auto-engine")
    doc = reader.read(pdf_path)
    
    print(f"Parsed with VLM: {len(doc.content)} characters")
    
    return doc


def with_ocr_language():
    """Example 4: Specify OCR language"""
    print("\n=== Example 4: Custom OCR Language ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Configure OCR language
    reader = MinerUReader(
        backend="pipeline",
        lang="en"  # or "ch" for Chinese, "ja" for Japanese, etc.
    )
    doc = reader.read(pdf_path)
    
    print(f"Parsed with English OCR: {len(doc.content)} characters")
    
    return doc


def parse_page_range():
    """Example 5: Parse specific page range"""
    print("\n=== Example 5: Parse Page Range ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Parse only pages 0-9 (0-based indexing)
    reader = MinerUReader(
        start_page_id=0,
        end_page_id=10
    )
    doc = reader.read(pdf_path)
    
    print(f"Parsed pages 1-10: {len(doc.pages)} pages")
    
    return doc


def disable_features():
    """Example 6: Disable formula or table parsing"""
    print("\n=== Example 6: Disable Features ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Disable formula and table recognition for faster parsing
    reader = MinerUReader(
        formula_enable=False,
        table_enable=False
    )
    doc = reader.read(pdf_path)
    
    print(f"Parsed without formula/table: {len(doc.content)} characters")
    
    return doc


def remote_vlm_service():
    """Example 7: Use remote VLM service"""
    print("\n=== Example 7: Remote VLM Service ===")
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    # Use remote VLM API service
    reader = MinerUReader(
        backend="hybrid-http-client",
        server_url="http://127.0.0.1:30000"
    )
    doc = reader.read(pdf_path)
    
    print(f"Parsed via remote service: {len(doc.content)} characters")
    
    return doc


def inspect_metadata(doc):
    """Example 8: Inspect document metadata"""
    print("\n=== Example 8: Document Metadata ===")
    
    print(f"Source: {doc.metadata.source}")
    print(f"File type: {doc.metadata.file_type}")
    print(f"Content length: {doc.metadata.content_length} characters")
    print(f"Reader: {doc.metadata.reader_name}")
    
    print("\nMinerU-specific metadata:")
    for key, value in doc.metadata.custom.items():
        print(f"  {key}: {value}")


def save_output(doc, output_path: str = "output.md"):
    """Example 9: Save parsed content"""
    print(f"\n=== Example 9: Save Output ===")
    
    # Save full document
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


if __name__ == "__main__":
    print("=" * 60)
    print("MinerUReader Examples")
    print("=" * 60)
    
    # Run examples (uncomment the ones you want to try)
    doc = basic_usage()
    # doc = cpu_only_pipeline()
    # doc = vlm_backend()
    # doc = with_ocr_language()
    # doc = parse_page_range()
    # doc = disable_features()
    # doc = remote_vlm_service()
    # inspect_metadata(doc)
    # save_output(doc)
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
