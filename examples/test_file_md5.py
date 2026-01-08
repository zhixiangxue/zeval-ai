"""
Test file MD5 computation and deduplication
"""

from zeval.synthetic_data.readers.base import BaseReader

def main():
    # Test file
    file_path = "tmp/Thunderbird Product Overview 2025 - No Doc.pdf"
    
    print("=" * 60)
    print("Testing File MD5 Computation")
    print("=" * 60)
    print()
    
    # Compute MD5
    print(f"Computing MD5 for: {file_path}")
    md5_hash = BaseReader.md5(file_path)
    print(f"MD5: {md5_hash}")
    print()

if __name__ == "__main__":
    main()
