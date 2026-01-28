"""
Docling reader for PDF understanding
"""

from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
from docling.pipeline.vlm_pipeline import VlmPipeline

from .base import BaseReader
from ...schemas.document import BaseDocument, Page
from ...schemas.metadata import DocumentMetadata
from ...schemas.pdf import PDF
from ...utils.source import SourceUtils, FileType, SourceInfo


class DoclingReader(BaseReader):
    """
    Reader using Docling library for PDF understanding
    
    Returns PDF document with:
    - content: Full markdown representation of the entire document
    - pages: List of Page objects, each containing that page's markdown text
    
    Usage:
        # Basic usage - default standard pipeline
        reader = DoclingReader()
        doc = reader.read("path/to/file.pdf")
        
        # Custom PDF pipeline options
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        
        pdf_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True
        )
        reader = DoclingReader(pdf_pipeline_options=pdf_options)
        
        # Use VLM pipeline
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        
        vlm_options = VlmPipelineOptions(...)
        reader = DoclingReader(vlm_pipeline_options=vlm_options)
        
        # GPU acceleration
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        
        pdf_options = PdfPipelineOptions()
        pdf_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA  # or AUTO, CPU, MPS
        )
        reader = DoclingReader(pdf_pipeline_options=pdf_options)
    """
    
    @staticmethod
    def _get_supported_types() -> set[FileType]:
        """
        Get supported file types
        
        Note: All formats except PDF use Docling's default pipeline.
        Only PDF format can be customized via pdf_pipeline_options or vlm_pipeline_options.
        """
        return {
            FileType.PDF,
            FileType.WORD,
            FileType.POWERPOINT,
            FileType.EXCEL,
            FileType.HTML,
        }
    
    def __init__(
        self,
        pdf_pipeline_options: Optional[PdfPipelineOptions] = None,
        vlm_pipeline_options: Optional[VlmPipelineOptions] = None,
    ):
        """
        Initialize Docling reader
        
        Args:
            pdf_pipeline_options: Options for standard PDF pipeline
            vlm_pipeline_options: Options for VLM pipeline
            
        Pipeline selection logic:
            - If vlm_pipeline_options is provided, use VLM pipeline
            - Otherwise, use standard PDF pipeline (with pdf_pipeline_options if provided)
            - If both are provided, VLM pipeline takes precedence
        """
        self._pdf_pipeline_options = pdf_pipeline_options
        self._vlm_pipeline_options = vlm_pipeline_options
        
        # Build format options
        format_options = self._build_format_options()
        
        # Initialize converter
        self._converter = DocumentConverter(format_options=format_options)
    
    def _build_format_options(self) -> dict[InputFormat, PdfFormatOption]:
        """
        Build format options for DocumentConverter
        
        Note:
            - Only PDF format is configured here with custom pipeline options
            - Other formats (WORD, POWERPOINT, EXCEL, HTML, IMAGE) use Docling's 
              default pipeline, which is the recommended approach per official examples
            - This matches the official Docling multi-format conversion pattern
        
        Returns:
            Dict of format options (currently only PDF is customized)
        """
        # If VLM options provided, use VLM pipeline
        if self._vlm_pipeline_options is not None:
            return {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self._vlm_pipeline_options,
                )
            }
        else:
            # Use standard PDF pipeline (with custom options if provided)
            return {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self._pdf_pipeline_options,
                )
            }
    
    def read(self, source: str) -> BaseDocument:
        """
        Read and convert a file to PDF document
        
        Args:
            source: File path (relative/absolute) or URL
            
        Returns:
            PDF document with markdown content and page list
            
        Raises:
            ValueError: If source is invalid or file format is not supported
        """
        # Validate and get complete info
        info = SourceUtils.validate(source, check_accessibility=True, timeout=5)
        
        # Check if valid
        if not info.is_valid:
            raise ValueError(info.error_message)
        
        # Check if file type is supported
        if info.file_type not in self._get_supported_types():
            raise ValueError(
                f"Unsupported file type: {info.file_type.value}. "
                f"Supported: {', '.join(t.value for t in self._get_supported_types())}"
            )
        
        # Convert file using Docling
        result = self._converter.convert(source)
        docling_doc = result.document
        
        # Extract markdown content
        markdown_content = docling_doc.export_to_markdown()
        
        # Build pages (each page contains its markdown text)
        pages = self._extract_pages(docling_doc)
        
        # Build metadata
        metadata = self._build_metadata(info, markdown_content, len(pages))
        
        # Create PDF document
        return PDF(
            content=markdown_content,
            metadata=metadata,
            pages=pages
        )
    
    def _extract_pages(self, docling_doc) -> list[Page]:
        """
        Extract pages from DoclingDocument
        
        Simply exports each page's Markdown content using Docling's native export.
        No complex manual grouping - just use the library's capability.
        
        Args:
            docling_doc: The DoclingDocument object
            
        Returns:
            List of Page objects with Markdown content
        """
        pages = []
        
        # Get page count from document
        # Note: doc.pages is a dict in Docling, keys are page numbers
        if hasattr(docling_doc, 'pages') and docling_doc.pages:
            page_numbers = sorted(docling_doc.pages.keys())
        else:
            # Fallback: try to infer from items
            page_numbers = set()
            for item in docling_doc.texts:
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_numbers.add(prov.page_no)
            page_numbers = sorted(page_numbers) if page_numbers else [1]
        
        # Create Page objects using Docling's native export
        for page_num in page_numbers:
            # Export page content using Docling's native export
            page_content = docling_doc.export_to_markdown(page_no=page_num)
            
            pages.append(Page(
                page_number=page_num,
                content=page_content,
                metadata={}
            ))
        
        return pages
    
    def _build_metadata(self, info: SourceInfo, content: str, page_count: int) -> DocumentMetadata:
        """
        Build metadata object from SourceInfo
        
        Args:
            info: SourceInfo from validation
            content: Document content
            page_count: Number of pages
            
        Returns:
            DocumentMetadata object
        """
        # Get file size if local file
        file_size = None
        if info.source_type.value == "local":
            try:
                file_size = Path(info.source).stat().st_size
            except Exception:
                pass
        
        return DocumentMetadata(
            source=info.source,
            source_type=info.source_type.value,
            file_type=info.file_type.value if info.file_type else "unknown",
            file_name=Path(info.source).name if info.source_type.value == "local" else None,
            file_size=file_size,
            file_extension=info.file_extension if info.file_extension else None,
            content_length=len(content),
            mime_type=info.mime_type if info.mime_type else None,
            reader_name="DoclingReader",
            custom={"page_count": page_count}
        )
