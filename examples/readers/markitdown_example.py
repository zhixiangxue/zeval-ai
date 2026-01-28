"""
MarkItDownReader Example

Demonstrates usage of MarkItDownReader for converting various document formats.
MarkItDown is simpler than Docling but supports more file types.
"""

from pathlib import Path
from zeval.synthetic_data.readers.markitdown import MarkItDownReader


def basic_usage():
    """Example 1: Basic usage with PDF"""
    print("\n=== Example 1: Convert PDF ===")
    
    # Get file path from user
    pdf_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    reader = MarkItDownReader()
    doc = reader.read(pdf_path)
    
    print(f"Document type: {type(doc).__name__}")  # PDF
    print(f"Content length: {len(doc.content)} characters")
    print(f"Number of pages: {len(doc.pages)}")  # Will be 0 - MarkItDown doesn't support pagination
    
    # Preview content
    print(f"\nContent preview:\n{doc.content[:500]}...")
    
    return doc


def convert_word_document():
    """Example 2: Convert Word document"""
    print("\n=== Example 2: Convert Word Document ===")
    
    # Get file path from user
    docx_path = input("Enter DOCX file path: ").strip().strip('"').strip("'")
    
    reader = MarkItDownReader()
    doc = reader.read(docx_path)
    
    print(f"Document type: {type(doc).__name__}")  # Markdown
    print(f"Content length: {len(doc.content)} characters")
    
    return doc


def convert_powerpoint():
    """Example 3: Convert PowerPoint"""
    print("\n=== Example 3: Convert PowerPoint ===")
    
    # Get file path from user
    pptx_path = input("Enter PPTX file path: ").strip().strip('"').strip("'")
    
    reader = MarkItDownReader()
    doc = reader.read(pptx_path)
    
    print(f"Document type: {type(doc).__name__}")  # Markdown
    print(f"Content length: {len(doc.content)} characters")
    print(f"\nContent preview:\n{doc.content[:300]}...")
    
    return doc


def convert_excel():
    """Example 4: Convert Excel"""
    print("\n=== Example 4: Convert Excel ===")
    
    # Get file path from user
    xlsx_path = input("Enter XLSX file path: ").strip().strip('"').strip("'")
    
    reader = MarkItDownReader()
    doc = reader.read(xlsx_path)
    
    print(f"Converted Excel to markdown")
    print(f"Content length: {len(doc.content)} characters")
    
    return doc


def convert_html():
    """Example 5: Convert HTML"""
    print("\n=== Example 5: Convert HTML ===")
    
    # Get file path from user
    html_path = input("Enter HTML file path: ").strip().strip('"').strip("'")
    
    reader = MarkItDownReader()
    doc = reader.read(html_path)
    
    print(f"Converted HTML to markdown")
    print(f"Content length: {len(doc.content)} characters")
    
    return doc


def batch_convert_files():
    """Example 6: Batch convert multiple files"""
    print("\n=== Example 6: Batch Convert ===")
    
    reader = MarkItDownReader()
    
    files = [
        "document1.pdf",
        "document2.docx",
        "presentation.pptx",
        "data.xlsx"
    ]
    
    results = []
    for file_path in files:
        try:
            doc = reader.read(file_path)
            results.append({
                "file": file_path,
                "type": type(doc).__name__,
                "length": len(doc.content),
                "success": True
            })
            print(f"✓ {file_path}: {len(doc.content)} chars")
        except Exception as e:
            results.append({
                "file": file_path,
                "error": str(e),
                "success": False
            })
            print(f"✗ {file_path}: {e}")
    
    return results


def save_output(doc, output_path: str = "output.md"):
    """Example 7: Save converted content"""
    print(f"\n=== Example 7: Save Output ===")
    
    output_file = Path(output_path)
    output_file.write_text(doc.content, encoding="utf-8")
    print(f"Saved to: {output_file.absolute()}")


def inspect_metadata(doc):
    """Example 8: Inspect metadata"""
    print("\n=== Example 8: Document Metadata ===")
    
    print(f"Source: {doc.metadata.source}")
    print(f"File type: {doc.metadata.file_type}")
    print(f"File name: {doc.metadata.file_name}")
    print(f"File size: {doc.metadata.file_size} bytes")
    print(f"Content length: {doc.metadata.content_length} characters")
    print(f"Reader: {doc.metadata.reader_name}")


if __name__ == "__main__":
    print("=" * 60)
    print("MarkItDownReader Examples")
    print("=" * 60)
    
    # Run examples (uncomment the ones you want to try)
    doc = basic_usage()
    # doc = convert_word_document()
    # doc = convert_powerpoint()
    # doc = convert_excel()
    # doc = convert_html()
    # results = batch_convert_files()
    # save_output(doc)
    # inspect_metadata(doc)
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
