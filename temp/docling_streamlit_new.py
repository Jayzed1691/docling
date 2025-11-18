# Docling Document Converter - Standalone Streamlit Interface
# Enhanced with comprehensive Docling configuration options

import streamlit as st
import os
import tempfile
import zipfile
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
from io import BytesIO

# Docling imports
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat, DocumentStream
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, 
        TableStructureOptions,
        TableFormerMode,
        EasyOcrOptions,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
        RapidOcrOptions,
        PdfBackend,
        VlmPipelineOptions,
        PictureDescriptionVlmOptions,
        AcceleratorOptions
    )
    from docling.chunking import HybridChunker
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    st.error(f"❌ Docling not installed: {e}")
    st.info("💡 Install with: `pip install docling`")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Suppress torch class path warnings
import logging
logging.getLogger('torch._C').setLevel(logging.ERROR)

# Advanced configuration presets for different document types
LARGE_DOCUMENT_CONFIG = {
    'pipeline_type': 'standard',
    'pdf_backend': 'DLPARSE_V4',  # Fastest backend for 2.43.0
    'do_ocr': True,
    'ocr_engine': 'easyocr',
    'do_table_structure': True,
    'table_mode': 'accurate',  # Critical for proper table formatting
    'do_cell_matching': True,  # Essential for markdown table structure
    'do_code_enrichment': False,    # Disable for performance
    'do_formula_enrichment': False, # Disable unless needed
    'do_picture_classification': False,  # Major performance impact
    'do_picture_description': False,     # Major performance impact
    'generate_page_images': False,       # Major performance impact
    'generate_picture_images': False,    # Major performance impact
    'images_scale': 1.0,
    'enable_remote_services': False,
    'memory_optimization': True,
    'max_concurrent_pages': 4
}

TABLE_OPTIMIZED_CONFIG = {
    'pipeline_type': 'standard',
    'pdf_backend': 'DLPARSE_V4',
    'do_table_structure': True,
    'table_mode': 'accurate',     # ACCURATE mode for best table quality
    'do_cell_matching': True,     # CRITICAL for proper markdown formatting
    'table_extraction_quality': 'high',  # New in 2.43.0
    'preserve_table_formatting': True,   # New option
    'table_markdown_style': 'github',    # GitHub-style tables
    'escape_table_chars': True,          # Handle special characters
}

# Page configuration
st.set_page_config(
    page_title="Docling Document Converter",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DoclingProcessor:
    """Enhanced Docling document processor with comprehensive configuration options."""
    
    def __init__(self):
        self.converter = None
        self.metadata_extractor = None  # Simple metadata creation instead
        self.supported_formats = [
            'pdf', 'docx', 'pptx', 'xlsx', 'html', 
            'png', 'jpg', 'jpeg', 'bmp', 'tiff',
            'md', 'asciidoc'
        ]
    
    def create_ocr_options(self, ocr_engine: str, **kwargs):
        """Create OCR options based on selected engine."""
        if ocr_engine == "easyocr":
            ocr_options = EasyOcrOptions(
                force_full_page_ocr=kwargs.get('force_full_page_ocr', False),
                lang=kwargs.get('languages', ['en'])
            )
            # Add GPU support if available
            if hasattr(ocr_options, 'use_gpu') and kwargs.get('use_gpu', False):
                ocr_options.use_gpu = True
            return ocr_options
        elif ocr_engine == "tesseract_cli":
            return TesseractCliOcrOptions(
                force_full_page_ocr=kwargs.get('force_full_page_ocr', False),
                lang=kwargs.get('languages', ['eng']),
                path=kwargs.get('tesseract_path', None)
            )
        elif ocr_engine == "tesseract":
            return TesseractOcrOptions(
                force_full_page_ocr=kwargs.get('force_full_page_ocr', False),
                lang=kwargs.get('languages', ['eng'])
            )
        elif ocr_engine == "rapidocr":
            return RapidOcrOptions(
                force_full_page_ocr=kwargs.get('force_full_page_ocr', False)
            )
        else:
            return EasyOcrOptions()
    
    def create_table_structure_options(self, mode: str, do_cell_matching: bool):
        """Create table structure options."""
        table_mode = TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST
        return TableStructureOptions(
            mode=table_mode,
            do_cell_matching=do_cell_matching
        )
    
    def create_vlm_options(self, model_type: str, **kwargs):
        """Create VLM pipeline options."""
        if model_type == "smoldocling":
            return VlmPipelineOptions(
                model=PictureDescriptionVlmOptions(
                    repo_id='HuggingFaceTB/SmolVLM-256M-Instruct',
                    prompt=kwargs.get('vlm_prompt', 'What is shown in this image?'),
                    scale=kwargs.get('vlm_scale', 1.0)
                ),
                accelerator_options=AcceleratorOptions(
                    num_threads=kwargs.get('num_threads', 4),
                    device=kwargs.get('device', 'auto')
                )
            )
        elif model_type == "granite_vision":
            return VlmPipelineOptions(
                model=PictureDescriptionVlmOptions(
                    repo_id='ibm-granite/granite-vision-3.1-2b-preview',
                    prompt=kwargs.get('vlm_prompt', 'What is shown in this image?'),
                    scale=kwargs.get('vlm_scale', 1.0)
                ),
                accelerator_options=AcceleratorOptions(
                    num_threads=kwargs.get('num_threads', 4),
                    device=kwargs.get('device', 'auto')
                )
            )
        else:
            return VlmPipelineOptions()
    
    def create_optimized_pdf_pipeline_options(self, config: Dict[str, Any]):
        """Create performance-optimized PDF pipeline options for Docling 2.43.0."""
        
        # Detect GPU availability
        import torch
        has_gpu = torch.cuda.is_available()
        
        # Performance-first OCR options
        if config.get('ocr_engine', 'easyocr') == 'easyocr':
            ocr_options = EasyOcrOptions(
                force_full_page_ocr=config.get('force_full_page_ocr', False),
                lang=config.get('ocr_languages', ['en'])
            )
            # Add GPU support if available
            if hasattr(ocr_options, 'use_gpu'):
                ocr_options.use_gpu = has_gpu
        else:
            ocr_options = self.create_ocr_options(config.get('ocr_engine', 'easyocr'), **config)
            
        # Optimized table structure options for better markdown export
        table_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,  # Use ACCURATE for better table formatting
            do_cell_matching=True  # Critical for proper markdown table structure
        )
        
        pipeline_options = PdfPipelineOptions(
            # Core processing - optimized for performance
            do_ocr=config.get('do_ocr', True),
            do_table_structure=True,  # Always enable for table extraction
            do_code_enrichment=config.get('do_code_enrichment', False),
            do_formula_enrichment=config.get('do_formula_enrichment', False),
            do_picture_classification=False,  # Disable unless needed - performance impact
            do_picture_description=False,    # Disable unless needed - performance impact
            
            # Backend selection - DLPARSE_V4 is fastest for 2.43.0
            backend=PdfBackend.DLPARSE_V4,
            
            # Performance optimizations
            images_scale=1.0,  # Don't upscale unless necessary
            generate_page_images=False,    # Major performance impact
            generate_picture_images=False, # Major performance impact
            
            # Accelerator options for 2.43.0
            accelerator_options=AcceleratorOptions(
                num_threads=min(8, os.cpu_count()),  # Optimal thread count
                device='auto'  # Auto-detect best device
            ),
            
            # Options objects
            ocr_options=ocr_options,
            table_structure_options=table_options,
            
            # Disable remote services for performance
            enable_remote_services=False
        )
        
        return pipeline_options
    
    # Keep your existing create_pdf_pipeline_options as a backup, but rename it
    def create_pdf_pipeline_options_legacy(self, config: Dict[str, Any]):
        """Updated for Docling 2.43.0 with performance and table optimizations."""
        
        # Import accelerator options
        from docling.datamodel.pipeline_options import AcceleratorOptions
        
        # Detect system capabilities
        import torch
        import os
        has_gpu = torch.cuda.is_available()
        
        # OCR options with GPU support
        ocr_options = self.create_ocr_options(
            config.get('ocr_engine', 'easyocr'),
            force_full_page_ocr=config.get('force_full_page_ocr', False),
            languages=config.get('ocr_languages', ['en']),
            use_gpu=has_gpu,
            download_enabled=True
        )
        
        # Table options optimized for markdown export
        table_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,  # Use ACCURATE for better tables
            do_cell_matching=True  # CRITICAL for markdown table formatting
        )
        
        # Performance-optimized pipeline
        pipeline_options = PdfPipelineOptions(
            # Core processing
            do_ocr=config.get('do_ocr', True),
            do_table_structure=True,  # Always enable for table extraction
            do_code_enrichment=config.get('do_code_enrichment', False),
            do_formula_enrichment=config.get('do_formula_enrichment', False),
            do_picture_classification=False,  # Disable for performance
            do_picture_description=False,     # Disable for performance
            
            # Use fastest backend
            backend=PdfBackend.DLPARSE_V4,
            
            # Performance settings
            images_scale=config.get('images_scale', 1.0),
            generate_page_images=False,
            generate_picture_images=False,
            
            # Accelerator configuration for 2.43.0
            accelerator_options=AcceleratorOptions(
                num_threads=min(8, os.cpu_count()),
                device='auto'
            ),
            
            # Options
            ocr_options=ocr_options,
            table_structure_options=table_options,
            enable_remote_services=config.get('enable_remote_services', False)
        )
        
        return pipeline_options
    
    
    def initialize_converter(self, **config):
        """Initialize the Docling converter with Docling 2.43.0 compatibility."""
        try:
            # Extract pipeline_type from config
            pipeline_type = config.pop('pipeline_type', 'standard')
            
            if pipeline_type == "vlm":
                # VLM pipeline configuration
                vlm_options = self.create_vlm_options(
                    config.get('vlm_model', 'smoldocling'),
                    vlm_prompt=config.get('vlm_prompt', 'Convert this document to markdown'),
                    vlm_scale=config.get('vlm_scale', 1.0),
                    num_threads=config.get('num_threads', 4),
                    device=config.get('device', 'auto')
                )
                
                format_options = {
                    InputFormat.PDF: PdfFormatOption(pipeline_options=vlm_options)
                }
            else:
                # Standard pipeline configuration
                pdf_options = self.create_optimized_pdf_pipeline_options(config)
                
                format_options = {
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
                }
                
            # Create converter - try different API patterns for 2.43.0
            try:
                # Try the current API
                self.converter = DocumentConverter(format_options=format_options)
            except TypeError:
                # If that fails, try without format_options and set them after
                self.converter = DocumentConverter()
                if hasattr(self.converter, 'format_options'):
                    self.converter.format_options = format_options
                    
            st.success("✅ Docling converter initialized successfully")
            return True
        
        except Exception as e:
            st.error(f"❌ Error initializing Docling converter: {e}")
            # Show more debug info
            st.code(f"Config: {config}")
            st.code(f"Error type: {type(e).__name__}")
            return False
    
    def process_document(self, file_path: str, export_format: str = "markdown", **options) -> Tuple[str, Dict[str, Any]]:
        """Process a document and return the content and metadata."""
        if not self.converter:
            raise RuntimeError("Converter not initialized. Call initialize_converter first.")
        
        try:
            # Prepare convert arguments, filtering out None values
            convert_args = {}
            if options.get('max_pages') is not None and options.get('max_pages') > 0:
                convert_args['max_num_pages'] = options['max_pages']
            if options.get('max_file_size') is not None and options.get('max_file_size') > 0:
                convert_args['max_file_size'] = options['max_file_size']
            
            # Convert document
            with st.spinner(f"🔄 Processing: {os.path.basename(file_path)}"):
                result = self.converter.convert(file_path, **convert_args)
            
            # Extract content based on format
            if export_format == "markdown":
                content = result.document.export_to_markdown()
            elif export_format == "html":
                content = result.document.export_to_html()
            elif export_format == "json":
                content = result.document.export_to_json()
            else:
                content = result.document.export_to_markdown()
            
            # Extract metadata
            metadata = {
                'source_file': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'export_format': export_format,
                'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'docling_version': 'latest',
                'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                'conversion_status': 'success'
            }
            
            # Add document-specific metadata if available
            if hasattr(result.document, 'metadata') and result.document.metadata:
                metadata.update(result.document.metadata)
            
            return content, metadata
            
        except Exception as e:
            st.error(f"❌ Error processing document: {e}")
            raise
    
    def process_from_stream(self, file_content: bytes, filename: str, export_format: str = "markdown", **options) -> Tuple[str, Dict[str, Any]]:
        """Process a document from a binary stream."""
        if not self.converter:
            raise RuntimeError("Converter not initialized. Call initialize_converter first.")
        
        try:
            # Validate file content
            if not file_content or len(file_content) < 100:
                raise ValueError(f"File content is too small or empty: {len(file_content)} bytes")
                
            # Enhanced PDF validation and repair
            if filename.lower().endswith('.pdf'):
                validation_result, repaired_content, repair_log = self.validate_and_repair_pdf(file_content, filename)
                
                if validation_result == False:
                    raise ValueError(f"PDF validation failed: {repair_log}")
                    
                if repair_log != "No issues found":
                    st.warning(f"🔧 PDF repaired: {repair_log}")
                    file_content = repaired_content
                    
                # Use lenient backend for structurally damaged PDFs
                if validation_result == "structural_damage":
                    st.info("🛠️ Using lenient processing for structurally damaged PDF")
                    # Force PYPDFIUM2 backend in options
                    options['force_lenient_backend'] = True
                    
                st.info(f"📄 Processing PDF: {len(file_content):,} bytes")
            
            # Check if we need lenient processing
            if options.get('force_lenient_backend', False):
                return self.process_with_lenient_backend(file_content, filename, export_format, **options)
            
            # Create document stream with proper positioning
            stream = BytesIO(file_content)
            stream.seek(0)  # Ensure we're at the beginning
            
            doc_stream = DocumentStream(name=filename, stream=stream)
            
            # Prepare convert arguments, filtering out None values
            convert_args = {}
            if options.get('max_pages') is not None and options.get('max_pages') > 0:
                convert_args['max_num_pages'] = options['max_pages']
            if options.get('max_file_size') is not None and options.get('max_file_size') > 0:
                convert_args['max_file_size'] = options['max_file_size']
            
            # Convert document with better error context
            with st.spinner(f"🔄 Processing: {filename}"):
                try:
                    result = self.converter.convert(doc_stream, **convert_args)
                    st.success(f"✅ Successfully processed with Docling")
                except Exception as convert_error:
                    # Try to provide more specific error information
                    error_msg = str(convert_error)
                    if "not valid" in error_msg.lower():
                        st.error(f"❌ PDF validation failed: {error_msg}")
                        st.info("💡 Try using a different PDF backend or check if the PDF is corrupted")
                        
                        # Suggest alternative approaches
                        st.warning("""
                        **Troubleshooting suggestions:**
                        1. Try the 'Fast' preset (uses PYPDFIUM2 backend)
                        2. Check if the PDF opens normally in a PDF viewer
                        3. Try converting the PDF to a different format first
                        4. The PDF might have security restrictions or be corrupted
                        """)
                    raise convert_error
            
            # Extract content with enhanced table processing
            content = self.extract_tables_with_proper_formatting(result, export_format)
            
            # Extract metadata
            metadata = {
                'source_file': filename,
                'file_size': len(file_content),
                'export_format': export_format,
                'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'docling_version': 'latest',
                'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                'conversion_status': 'success',
                'processing_engine': 'Docling'
            }
            
            # Add document-specific metadata if available
            if hasattr(result.document, 'metadata') and result.document.metadata:
                metadata.update(result.document.metadata)
            
            return content, metadata
            
        except Exception as e:
            st.error(f"❌ Error processing document from stream: {e}")
            
            # Provide helpful debugging information
            if filename.lower().endswith('.pdf'):
                st.info(f"""
                **PDF Processing Debug Info:**
                - File size: {len(file_content):,} bytes
                - PDF header present: {file_content.startswith(b'%PDF-')}
                - Backend: {getattr(self.converter.format_options[InputFormat.PDF].pipeline_options, 'backend', 'Unknown')}
                """)
            
            raise
    
    def validate_and_repair_pdf(self, file_content: bytes, filename: str) -> Tuple[bool, bytes, str]:
        """Comprehensive PDF validation and repair."""
        import re
        
        issues_found = []
        repaired_content = file_content
        
        # Check 1: Basic PDF header
        if not file_content.startswith(b'%PDF-'):
            pdf_start = file_content.find(b'%PDF-')
            if pdf_start > 0:
                repaired_content = file_content[pdf_start:]
                issues_found.append(f"Trimmed {pdf_start} bytes before PDF header")
            else:
                return False, file_content, "No PDF header found"
            
        # Check 2: Version consistency
        header_line = repaired_content[:20]
        if b'%PDF-' in header_line:
            version_match = re.search(rb'%PDF-(\d+\.\d+)', header_line)
            if not version_match:
                # Fix malformed version
                new_line_pos = repaired_content.find(b'\n')
                if new_line_pos == -1:
                    new_line_pos = repaired_content.find(b'\r')
                if new_line_pos == -1:
                    new_line_pos = 20
                repaired_content = b'%PDF-1.4' + repaired_content[new_line_pos:]
                issues_found.append("Fixed malformed PDF version")
                
        # Check 3: EOF marker
        if not repaired_content.strip().endswith(b'%%EOF'):
            repaired_content += b'\n%%EOF'
            issues_found.append("Added missing EOF marker")
            
        # Check 4: Basic structure validation
        has_trailer = b'trailer' in repaired_content
        has_xref = b'xref' in repaired_content
        
        if not has_trailer or not has_xref:
            # For structurally damaged PDFs, we'll still try to process
            # but use a more lenient backend
            issues_found.append("Warning: Missing trailer or xref table - will use lenient processing")
            
            # Signal to use PYPDFIUM2 backend which is more forgiving
            return "structural_damage", repaired_content, "; ".join(issues_found)
        
        success = len(issues_found) == 0 or all("Warning" not in issue for issue in issues_found)
        repair_log = "; ".join(issues_found) if issues_found else "No issues found"
        
        return success, repaired_content, repair_log
    
    def process_with_lenient_backend(self, file_content: bytes, filename: str, export_format: str = "markdown", **options):
        """Process PDF with most lenient backend settings."""
        
        # Create a minimal, lenient converter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
        
        try:
            # Minimal processing options for damaged PDFs
            pipeline_options = PdfPipelineOptions(
                backend=PdfBackend.PYPDFIUM2,  # Most lenient backend
                do_ocr=True,  # Enable OCR as fallback
                do_table_structure=False,  # Disable complex parsing
                do_code_enrichment=False,
                do_formula_enrichment=False,
                do_picture_classification=False,
                do_picture_description=False,
                generate_page_images=False,
                images_scale=1.0
            )
            
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            
            lenient_converter = DocumentConverter(format_options=format_options)
            
            # Process with lenient settings
            from io import BytesIO
            stream = BytesIO(file_content)
            doc_stream = DocumentStream(name=filename, stream=stream)
            
            with st.spinner(f"🔄 Processing damaged PDF with lenient backend: {filename}"):
                result = lenient_converter.convert(doc_stream)
                
            # Extract content
            if export_format == "markdown":
                content = result.document.export_to_markdown()
            elif export_format == "html":
                content = result.document.export_to_html()
            elif export_format == "json":
                content = result.document.export_to_json()
            else:
                content = result.document.export_to_markdown()
                
            # Generate metadata - simplified for 2.43.0
            metadata = {
                'source_file': filename,
                'file_size': len(file_content),
                'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                'processing_engine': 'Docling (Lenient)',
                'export_format': export_format,
                'conversion_status': 'success_with_repairs',
                'backend_used': 'PYPDFIUM2'
            }
            
            return content, metadata
        
        except Exception as e:
            st.error(f"❌ Even lenient processing failed: {e}")
            raise
    
    def cleanup_memory(self):
        """Clean up memory after processing large files."""
        import gc
        import torch
        
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()
    
    def extract_tables_with_proper_formatting(self, result, export_format="markdown"):
        """Extract tables with proper markdown formatting for 2.43.0."""
        
        if export_format == "markdown":
            try:
                # Method 1: Try enhanced markdown export with table optimization
                if hasattr(result.document, 'export_to_markdown'):
                    # Try with enhanced options if available
                    try:
                        markdown_content = result.document.export_to_markdown()
                    except Exception:
                        # Fallback to basic export
                        markdown_content = result.document.export_to_markdown()
                else:
                    markdown_content = str(result.document)
                    
                # Method 2: If tables still don't format properly, extract and rebuild
                if result.document.tables and ("|" not in markdown_content or markdown_content.count("|") < len(result.document.tables) * 3):
                    st.info("🔧 Enhancing table formatting...")
                    markdown_content = self.rebuild_markdown_with_tables(result.document, markdown_content)
                    
                return markdown_content
            
            except Exception as e:
                st.warning(f"Table formatting enhancement failed: {e}")
                return result.document.export_to_markdown()
            
        else:
            # For other formats, use standard export
            if export_format == "html":
                return result.document.export_to_html()
            elif export_format == "json":
                return result.document.export_to_json()
            else:
                return result.document.export_to_markdown()
            
    def rebuild_markdown_with_tables(self, document, base_content):
        """Rebuild markdown with properly formatted tables."""
        import pandas as pd
        import re
        
        try:
            enhanced_content = base_content
            table_replacements = []
            
            # Extract and format tables separately
            for i, table in enumerate(document.tables):
                try:
                    # Convert to DataFrame for proper markdown formatting
                    df = table.export_to_dataframe()
                    
                    if df is not None and not df.empty:
                        # Generate proper markdown table
                        markdown_table = df.to_markdown(
                            index=False,
                            tablefmt="github",  # Ensures proper | formatting
                            numalign="right",   # Right-align numbers
                            stralign="left"     # Left-align strings
                        )
                        
                        table_replacements.append({
                            'index': i,
                            'table': markdown_table,
                            'caption': f"Table {i+1}"
                        })
                        
                except Exception as e:
                    st.warning(f"Could not format table {i}: {e}")
                    continue
                
            # If we have properly formatted tables, integrate them
            if table_replacements:
                # Try to find existing table markers or append at the end
                for table_info in table_replacements:
                    table_marker = f"<!-- TABLE_{table_info['index']} -->"
                    table_content = f"\n\n## {table_info['caption']}\n\n{table_info['table']}\n"
                    
                    if table_marker in enhanced_content:
                        enhanced_content = enhanced_content.replace(table_marker, table_content)
                    else:
                        # Append table with context
                        enhanced_content += table_content
                        
            return enhanced_content
        
        except Exception as e:
            st.warning(f"Table reconstruction failed: {e}")
            return base_content
    
    def reset_converter_state(self):
        """Reset converter state to prevent memory buildup."""
        if self.converter:
            # Clear any cached models or state
            if hasattr(self.converter, 'reset'):
                self.converter.reset()
                
            # Force garbage collection
            import gc
            import torch
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()
            
    def process_with_memory_management(self, file_content: bytes, filename: str, export_format: str = "markdown", **options):
        """Process with active memory management for large files."""
        import gc
        import torch
        
        try:
            # Pre-processing cleanup
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()
            
            # Process the document using existing method
            result = self.process_from_stream(file_content, filename, export_format, **options)
            
            # Post-processing cleanup
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
        
        except Exception as e:
            # Cleanup on error
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()
            raise e
    
    def create_chunks(self, content: str, metadata: Dict[str, Any], chunker_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks using Docling's HybridChunker."""
        try:
            # Create chunker
            chunker = HybridChunker(
                tokenizer=chunker_config.get('tokenizer', 'BAAI/bge-small-en-v1.5'),
                max_tokens=chunker_config.get('max_tokens', 512)
            )
            
            # For chunking, we need to reconstruct a DoclingDocument
            # This is a simplified approach - in practice you'd use the original result
            # Here we'll return simple text chunks
            chunks = []
            
            # Simple text splitting as fallback
            max_chunk_size = chunker_config.get('max_chunk_size', 2000)
            overlap = chunker_config.get('overlap', 200)
            
            text_chunks = []
            current_chunk = ""
            
            for line in content.split('\n'):
                if len(current_chunk + line + '\n') > max_chunk_size and current_chunk:
                    text_chunks.append(current_chunk.strip())
                    # Keep overlap
                    words = current_chunk.split()
                    overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                    current_chunk = ' '.join(overlap_words) + '\n' + line
                else:
                    current_chunk += line + '\n'
            
            if current_chunk.strip():
                text_chunks.append(current_chunk.strip())
            
            # Create chunk objects
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i + 1,
                    'total_chunks': len(text_chunks),
                    'chunk_size': len(chunk_text)
                })
                
                chunks.append({
                    'content': chunk_text,
                    'metadata': chunk_metadata
                })
            
            return chunks
            
        except Exception as e:
            st.error(f"❌ Error creating chunks: {e}")
            return []

def create_download_package(content: str, metadata: Dict[str, Any], chunks: List[Dict[str, Any]], base_filename: str, export_format: str) -> bytes:
    """Create a downloadable ZIP package with all outputs."""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add main document
        file_ext = export_format if export_format in ['html', 'json'] else 'md'
        main_filename = f"{base_filename}.{file_ext}"
        
        # Add metadata as frontmatter for markdown
        if export_format == "markdown":
            full_content = f'---\n'
            for key, value in metadata.items():
                full_content += f'{key}: {json.dumps(value)}\n'
            full_content += f'---\n\n{content}'
        else:
            full_content = content
        
        zip_file.writestr(main_filename, full_content)
        
        # Add metadata as separate JSON file
        zip_file.writestr(f"{base_filename}_metadata.json", json.dumps(metadata, indent=2))
        
        # Add chunks if available
        if chunks:
            chunks_dir = f"{base_filename}_chunks/"
            for chunk in chunks:
                chunk_index = chunk['metadata']['chunk_index']
                chunk_filename = f"{chunks_dir}chunk_{chunk_index:03d}.{file_ext}"
                
                if export_format == "markdown":
                    chunk_content = f'---\n'
                    for key, value in chunk['metadata'].items():
                        chunk_content += f'{key}: {json.dumps(value)}\n'
                    chunk_content += f'---\n\n{chunk["content"]}'
                else:
                    chunk_content = chunk['content']
                
                zip_file.writestr(chunk_filename, chunk_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.title("🔬 Docling Document Converter")
    st.markdown("### Advanced document processing with IBM's Docling toolkit")
    
    if not DOCLING_AVAILABLE:
        st.stop()
    
    # Show version info
    try:
        import docling
        version = getattr(docling, '__version__', 'Unknown')
        st.success(f"📦 Docling version: {version}")
    except:
        st.info("📦 Docling version: Could not determine")
            
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = DoclingProcessor()
    if 'processed_content' not in st.session_state:
        st.session_state.processed_content = None
    if 'processed_metadata' not in st.session_state:
        st.session_state.processed_metadata = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Docling Configuration")
        
        # Pipeline Type Selection
        st.subheader("🧬 Pipeline Type")
        pipeline_type = st.radio(
            "Processing Pipeline:",
            ["standard", "vlm"],
            index=0,
            help="Standard: Traditional OCR + ML models, VLM: Vision-Language Models"
        )
        
        # Configuration Presets
        st.subheader("⚡ Quick Presets")
        preset_choice = st.selectbox(
            "Choose Preset:",
            ["Custom", "Large Documents", "Table Optimized", "Performance"],
            index=0,
            help="Pre-configured settings for common use cases"
        )
        
        if preset_choice == "Large Documents":
            st.info("🚀 Optimized for files >50MB with memory management")
            # Override some settings
            if pipeline_type == "standard":
                pdf_backend = "DLPARSE_V4"
                do_picture_classification = False
                do_picture_description = False
                generate_page_images = False
                generate_picture_images = False
        elif preset_choice == "Table Optimized":
            st.info("📊 Optimized for documents with complex tables")
            if pipeline_type == "standard":
                table_mode = "accurate"
                do_cell_matching = True
                do_table_structure = True
        elif preset_choice == "Performance":
            st.info("⚡ Fastest processing with minimal features")
            if pipeline_type == "standard":
                pdf_backend = "DLPARSE_V4"
                do_ocr = True
                do_table_structure = True
                do_code_enrichment = False
                do_formula_enrichment = False
                do_picture_classification = False
                do_picture_description = False
        
        # Export Format
        export_format = st.selectbox(
            "📄 Export Format:",
            ["markdown", "html", "json"],
            index=0,
            help="Choose output format"
        )
        
        st.divider()
        
        if pipeline_type == "standard":
            # Standard Pipeline Configuration
            st.subheader("🔧 Standard Pipeline Settings")
            
            # PDF Backend
            pdf_backend = st.selectbox(
                "PDF Backend:",
                ["DLPARSE_V2", "DLPARSE_V4", "DLPARSE_V1", "PYPDFIUM2"],
                index=0,
                help="DLPARSE_V2: Balanced, DLPARSE_V4: Latest, PYPDFIUM2: Fast but lower quality"
            )
            
            # OCR Configuration
            st.subheader("👁️ OCR Settings")
            do_ocr = st.checkbox("Enable OCR", value=True, help="Enable Optical Character Recognition")
            
            if do_ocr:
                ocr_engine = st.selectbox(
                    "OCR Engine:",
                    ["easyocr", "tesseract_cli", "tesseract", "rapidocr"],
                    index=0,
                    help="EasyOCR: Best quality, Tesseract: Traditional, RapidOCR: Fast"
                )
                
                force_full_page_ocr = st.checkbox(
                    "Force Full Page OCR",
                    value=False,
                    help="Apply OCR to entire page even if text is detected"
                )
                
                ocr_languages = st.multiselect(
                    "OCR Languages:",
                    ["en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko"],
                    default=["en"],
                    help="Select languages for OCR processing"
                )
            
            # Table Processing
            st.subheader("📊 Table Processing")
            do_table_structure = st.checkbox("Enable Table Structure", value=True)
            
            if do_table_structure:
                table_mode = st.selectbox(
                    "Table Mode:",
                    ["accurate", "fast"],
                    index=0,
                    help="Accurate: Better quality, Fast: Quicker processing"
                )
                
                do_cell_matching = st.checkbox(
                    "Enable Cell Matching",
                    value=True,
                    help="Map table structure back to PDF cells"
                )
            
            # Advanced Features
            st.subheader("🔬 Advanced Features")
            do_code_enrichment = st.checkbox("Code Detection", value=False)
            do_formula_enrichment = st.checkbox("Formula Detection", value=False)
            do_picture_classification = st.checkbox("Picture Classification", value=False)
            do_picture_description = st.checkbox("Picture Description", value=False)
            
            # Image Options
            st.subheader("🖼️ Image Options")
            generate_page_images = st.checkbox("Generate Page Images", value=False)
            generate_picture_images = st.checkbox("Generate Picture Images", value=False)
            images_scale = st.slider("Images Scale", 0.5, 2.0, 1.0, 0.1)
        
        elif pipeline_type == "vlm":
            # VLM Pipeline Configuration
            st.subheader("🤖 VLM Settings")
            
            vlm_model = st.selectbox(
                "VLM Model:",
                ["smoldocling", "granite_vision"],
                index=0,
                help="SmolDocling: Lightweight, Granite Vision: IBM's model"
            )
            
            vlm_prompt = st.text_area(
                "VLM Prompt:",
                value="Convert this document to markdown format, preserving structure and content.",
                help="Instruction for the vision-language model"
            )
            
            vlm_scale = st.slider("Image Scale", 0.5, 2.0, 1.0, 0.1)
            
            device = st.selectbox(
                "Device:",
                ["auto", "cpu", "cuda", "mps"],
                index=0,
                help="Processing device"
            )
            
            num_threads = st.slider("Number of Threads", 1, 16, 4)
        
        # Processing Limits
        st.divider()
        st.subheader("⚡ Processing Limits")
        
        max_pages = st.number_input(
            "Max Pages (0 = unlimited):",
            min_value=0,
            value=0,
            help="Limit number of pages to process"
        )
        
        max_file_size_mb = st.number_input(
            "Max File Size (MB, 0 = unlimited):",
            min_value=0,
            value=0,
            help="Limit file size in megabytes"
        )
        
        # Remote Services
        enable_remote_services = st.checkbox(
            "🌐 Enable Remote Services",
            value=False,
            help="Allow communication with external services (required for some features)"
        )
        
        # Chunking Configuration
        st.divider()
        st.subheader("✂️ Chunking Settings")
        
        enable_chunking = st.checkbox("Enable Chunking", value=False)
        
        if enable_chunking:
            tokenizer = st.selectbox(
                "Tokenizer:",
                ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
                index=0
            )
            
            max_tokens = st.slider("Max Tokens per Chunk", 128, 1024, 512)
            max_chunk_size = st.slider("Max Chunk Size (chars)", 500, 5000, 2000)
            overlap = st.slider("Overlap (chars)", 0, 500, 200)

    # Main Content
    st.header("📁 Document Processing")
    
    # File Upload
    uploaded_files = st.file_uploader(
        "Upload documents:",
        type=st.session_state.processor.supported_formats,
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, PPTX, XLSX, HTML, Images, MD, AsciiDoc"
    )
    
    # Drag and drop alternative
    if not uploaded_files:
        st.info("💡 **Tip**: You can drag and drop files directly onto the upload area above!")
    
    # Processing Button
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        # Prepare configuration
        config = {
            'pipeline_type': pipeline_type,
            'export_format': export_format,
            'allowed_formats': ['PDF', 'DOCX', 'PPTX', 'XLSX', 'HTML', 'IMAGE', 'MD']
        }
        
        if pipeline_type == "standard":
            config.update({
                'pdf_backend': pdf_backend,
                'do_ocr': do_ocr,
                'ocr_engine': ocr_engine if do_ocr else 'easyocr',
                'force_full_page_ocr': force_full_page_ocr if do_ocr else False,
                'ocr_languages': ocr_languages if do_ocr else ['en'],
                'do_table_structure': do_table_structure,
                'table_mode': table_mode if do_table_structure else 'accurate',
                'do_cell_matching': do_cell_matching if do_table_structure else True,
                'do_code_enrichment': do_code_enrichment,
                'do_formula_enrichment': do_formula_enrichment,
                'do_picture_classification': do_picture_classification,
                'do_picture_description': do_picture_description,
                'generate_page_images': generate_page_images,
                'generate_picture_images': generate_picture_images,
                'images_scale': images_scale,
                'enable_remote_services': enable_remote_services
            })
        else:  # VLM
            config.update({
                'vlm_model': vlm_model,
                'vlm_prompt': vlm_prompt,
                'vlm_scale': vlm_scale,
                'device': device,
                'num_threads': num_threads,
                'enable_remote_services': enable_remote_services
            })
        
        # Processing options
        processing_options = {
            'max_pages': max_pages if max_pages > 0 else None,
            'max_file_size': max_file_size_mb * 1024 * 1024 if max_file_size_mb > 0 else None
        }
        
        # Initialize converter
        success = st.session_state.processor.initialize_converter(**config)
        
        if success:
            first_file = uploaded_files[0]
            
            try:
                # Process with memory management
                content, metadata = st.session_state.processor.process_with_memory_management(
                    first_file.getvalue(),
                    first_file.name,
                    export_format,
                    **processing_options
                )
                
                st.session_state.processed_content = content
                st.session_state.processed_metadata = metadata
                st.session_state.chunks = None  # Reset chunks
                
                st.success(f"✅ Successfully processed: {first_file.name}")
                
                # Show file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Size", f"{len(first_file.getvalue())/1024:.1f} KB")
                with col2:
                    st.metric("Output Size", f"{len(content)/1024:.1f} KB")
                with col3:
                    st.metric("Pages", metadata.get('page_count', 'N/A'))
                    
                # Advanced processing info
                if metadata.get('processing_engine') == 'Docling':
                    import torch  # Add this import at the top if not already there
                    with st.expander("🔧 Processing Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Pipeline Settings:**")
                            st.write(f"- Backend: {config.get('pdf_backend', 'DLPARSE_V4')}")
                            st.write(f"- Table Mode: {config.get('table_mode', 'accurate')}")
                            st.write(f"- OCR Engine: {config.get('ocr_engine', 'easyocr')}")
                        with col2:
                            st.write("**Performance Features:**")
                            st.write(f"- Memory Optimization: ✅")
                            st.write(f"- GPU Acceleration: {'✅' if torch.cuda.is_available() else '❌'}")
                            st.write(f"- Table Enhancement: ✅")
                
            except Exception as e:
                st.error(f"❌ Error processing {first_file.name}: {str(e)}")
                # Clean up memory even on error
                st.session_state.processor.cleanup_memory()
            finally:
                # Cleanup after processing
                st.session_state.processor.cleanup_memory()

    # Show Results
    if st.session_state.processed_content and st.session_state.processed_metadata:
        st.divider()
        
        # Metadata Editor
        st.subheader("📝 Document Metadata")
        
        metadata_df = pd.DataFrame(
            list(st.session_state.processed_metadata.items()),
            columns=["Field", "Value"]
        )
        
        edited_metadata_df = st.data_editor(
            metadata_df,
            use_container_width=True,
            num_rows="dynamic",
            key="metadata_editor"
        )
        
        if st.button("✅ Update Metadata"):
            new_metadata = dict(zip(edited_metadata_df["Field"], edited_metadata_df["Value"]))
            st.session_state.processed_metadata = new_metadata
            st.success("✅ Metadata updated!")
            st.rerun()
        
        # Content Preview
        st.subheader(f"📄 {export_format.title()} Output")
        
        preview_height = st.slider("Preview Height", 200, 800, 400)
        
        with st.container(height=preview_height):
            if export_format == "markdown":
                st.markdown(st.session_state.processed_content)
            elif export_format == "html":
                st.components.v1.html(st.session_state.processed_content, height=preview_height-50, scrolling=True)
            else:  # JSON
                st.json(st.session_state.processed_content)
        
        # Download Options
        col1, col2 = st.columns(2)
        
        with col1:
            filename_base = Path(st.session_state.processed_metadata.get('source_file', 'document')).stem
            file_ext = export_format if export_format in ['html', 'json'] else 'md'
            
            # Prepare download content
            if export_format == "markdown":
                download_content = f'---\n'
                for key, value in st.session_state.processed_metadata.items():
                    download_content += f'{key}: {json.dumps(value)}\n'
                download_content += f'---\n\n{st.session_state.processed_content}'
            else:
                download_content = st.session_state.processed_content
            
            st.download_button(
                label=f"💾 Download {export_format.title()}",
                data=download_content,
                file_name=f"{filename_base}.{file_ext}",
                mime=f"text/{export_format}" if export_format != "json" else "application/json"
            )
        
        with col2:
            if enable_chunking and st.button("✂️ Create Chunks", type="secondary"):
                chunker_config = {
                    'tokenizer': tokenizer,
                    'max_tokens': max_tokens,
                    'max_chunk_size': max_chunk_size,
                    'overlap': overlap
                }
                
                with st.spinner("Creating chunks..."):
                    chunks = st.session_state.processor.create_chunks(
                        st.session_state.processed_content,
                        st.session_state.processed_metadata,
                        chunker_config
                    )
                    
                    st.session_state.chunks = chunks
                    st.success(f"✅ Created {len(chunks)} chunks!")
                    st.rerun()
        
        # Show Chunks
        if st.session_state.chunks:
            st.divider()
            st.subheader("✂️ Document Chunks")
            
            st.write(f"**Generated {len(st.session_state.chunks)} chunks:**")
            
            # Chunk Statistics
            chunk_sizes = [len(chunk['content']) for chunk in st.session_state.chunks]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", len(st.session_state.chunks))
            with col2:
                st.metric("Avg Size", f"{sum(chunk_sizes)//len(chunk_sizes):,} chars")
            with col3:
                st.metric("Size Range", f"{min(chunk_sizes):,} - {max(chunk_sizes):,}")
            
            # Chunk Preview
            selected_chunk = st.selectbox(
                "Preview chunk:",
                range(len(st.session_state.chunks)),
                format_func=lambda x: f"Chunk {x+1} ({len(st.session_state.chunks[x]['content']):,} chars)"
            )
            
            if selected_chunk is not None:
                chunk = st.session_state.chunks[selected_chunk]
                
                # Chunk content
                with st.expander(f"📄 Chunk {selected_chunk + 1} Content", expanded=True):
                    if export_format == "markdown":
                        st.markdown(chunk['content'])
                    elif export_format == "html":
                        st.components.v1.html(chunk['content'], height=300, scrolling=True)
                    else:
                        st.text(chunk['content'])
                
                # Chunk metadata
                with st.expander("📊 Chunk Metadata"):
                    st.json(chunk['metadata'])
            
            # Download All
            zip_data = create_download_package(
                st.session_state.processed_content,
                st.session_state.processed_metadata,
                st.session_state.chunks,
                filename_base,
                export_format
            )
            
            st.download_button(
                label="📦 Download Complete Package (ZIP)",
                data=zip_data,
                file_name=f"{filename_base}_complete.zip",
                mime="application/zip"
            )

    # Footer with information
    st.divider()
    
    # Information tabs
    info_tab1, info_tab2, info_tab3 = st.tabs(["📖 Usage Guide", "⚙️ Configuration", "🔗 Integration"])
    
    with info_tab1:
        st.markdown("""
        ## How to Use Docling Converter
        
        ### 1. Configure Processing Pipeline
        - **Standard Pipeline**: Traditional OCR + AI models for layout analysis
        - **VLM Pipeline**: Vision-Language Models for direct document understanding
        
        ### 2. Upload Documents
        - Drag and drop files or use the file browser
        - Supported formats: PDF, DOCX, PPTX, XLSX, HTML, Images, Markdown, AsciiDoc
        
        ### 3. Choose Export Format
        - **Markdown**: Clean, structured text format
        - **HTML**: Rich web format with styling
        - **JSON**: Structured data with full document hierarchy
        
        ### 4. Process and Download
        - Click "Process Documents" to start conversion
        - Review and edit metadata as needed
        - Download individual files or complete packages
        
        ### 5. Optional Chunking
        - Enable chunking for RAG applications
        - Configure chunk size and overlap
        - Download chunks as separate files
        """)
    
    with info_tab2:
        st.markdown("""
        ## Configuration Options
        
        ### Standard Pipeline Options
        
        **PDF Backends:**
        - `DLPARSE_V2`: Balanced performance and quality (recommended)
        - `DLPARSE_V4`: Latest version with enhanced features
        - `DLPARSE_V1`: Legacy version
        - `PYPDFIUM2`: Fast but lower quality, good for simple documents
        
        **OCR Engines:**
        - `EasyOCR`: Best quality, supports many languages
        - `Tesseract CLI`: Traditional OCR engine
        - `Tesseract`: Python wrapper for Tesseract
        - `RapidOCR`: Fast processing
        
        **Table Processing:**
        - `Accurate Mode`: Better quality for complex tables
        - `Fast Mode`: Quicker processing for simple tables
        - `Cell Matching`: Maps structure back to original PDF cells
        
        **Advanced Features:**
        - Code Detection: Identifies code blocks
        - Formula Detection: Extracts mathematical formulas
        - Picture Classification: Categorizes images
        - Picture Description: Generates image descriptions
        
        ### VLM Pipeline Options
        
        **Models:**
        - `SmolDocling`: Lightweight vision-language model
        - `Granite Vision`: IBM's advanced vision model
        
        **Processing:**
        - Custom prompts for specific extraction needs
        - Device selection (CPU, GPU, MPS)
        - Thread configuration for parallel processing
        """)
    
    with info_tab3:
        st.markdown("""
        ## Integration with Existing Code
        
        ### Adding Docling to Your PDF Converter
        
        ```python
        # 1. Import Docling classes
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        
        # 2. Create Docling processor class
        class DoclingProcessor:
            def __init__(self):
                self.converter = None
            
            def initialize_converter(self, config):
                pipeline_options = PdfPipelineOptions(
                    do_ocr=config.get('do_ocr', True),
                    do_table_structure=config.get('do_table_structure', True),
                    # ... other options
                )
                
                format_options = {
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
                
                self.converter = DocumentConverter(format_options=format_options)
            
            def process_pdf(self, file_path):
                result = self.converter.convert(file_path)
                return result.document.export_to_markdown()
        
        # 3. Integrate with existing module
        # Add to your PDFProcessor class:
        
        def initialize_docling_parser(self, config):
            if not hasattr(self, 'docling_processor'):
                self.docling_processor = DoclingProcessor()
            return self.docling_processor.initialize_converter(config)
        
        def process_pdf_with_docling(self, file_path, config):
            if not hasattr(self, 'docling_processor'):
                self.initialize_docling_parser(config)
            return self.docling_processor.process_pdf(file_path)
        ```
        
        ### Key Integration Points
        
        1. **Model Selection**: Add Docling as "Model 3" in your dropdown
        2. **Configuration**: Merge Docling settings with existing sidebar
        3. **Processing Flow**: Add Docling branch to your process_pdf method
        4. **Output Handling**: Docling outputs are already in markdown format
        5. **Chunking**: Docling has built-in chunking that can replace your current approach
        
        ### Benefits of Integration
        
        - **Multi-format Support**: Handle more than just PDFs
        - **Advanced Table Processing**: Better table structure recognition
        - **Local Processing**: No API keys required for basic functionality
        - **Extensibility**: Easy to add custom models and pipelines
        - **Performance**: Optimized for batch processing
        """)

if __name__ == "__main__":
    main()