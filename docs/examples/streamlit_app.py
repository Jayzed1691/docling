#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docling Document Converter - Enhanced Streamlit UI with Ollama Integration
===========================================================================
Features:
- Full Ollama API support for VLM models (qwen3-vl, granite-vision, etc.)
- Standard pipeline with OCR and table extraction
- PDF validation and repair
- Memory management for large documents
- Document chunking for RAG applications
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import warnings
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Docling imports
try:
    import docling
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        EasyOcrOptions,
        PdfBackend,
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TableStructureOptions,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
        VlmPipelineOptions,
    )
    from docling.datamodel.pipeline_options_vlm_model import (
        ApiVlmOptions,
        ResponseFormat,
    )
    from docling.datamodel.vlm_model_specs import (
        GRANITE_VISION_OLLAMA,
        GRANITE_VISION_TRANSFORMERS,
        QWEN3_VL_8B_OLLAMA,
        SMOLDOCLING_MLX,
        SMOLDOCLING_TRANSFORMERS,
        VlmModelType,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    st.error(f"Docling not installed: {e}")
    st.info("Install with: `pip install docling`")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("torch._C").setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="Docling Document Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_document_converter(
    _format_options: Dict[InputFormat, PdfFormatOption],
) -> DocumentConverter:
    """Return a cached DocumentConverter instance."""
    return DocumentConverter(format_options=_format_options)


class DoclingProcessor:
    """Enhanced Docling processor with Ollama VLM support."""

    def __init__(self):
        self.converter: Optional[DocumentConverter] = None
        self.supported_formats = [
            "pdf", "docx", "pptx", "xlsx", "html",
            "png", "jpg", "jpeg", "bmp", "tiff",
            "md", "asciidoc",
        ]

    @staticmethod
    def _create_ocr_options(ocr_engine: str, **kwargs) -> Any:
        """Create OCR options based on selected engine."""
        force = kwargs.get("force_full_page_ocr", False)
        langs = kwargs.get("languages", ["en"])

        if ocr_engine == "easyocr":
            ocr = EasyOcrOptions(force_full_page_ocr=force, lang=langs)
            if TORCH_AVAILABLE and hasattr(ocr, "use_gpu"):
                ocr.use_gpu = torch.cuda.is_available()
            return ocr
        elif ocr_engine == "tesseract_cli":
            return TesseractCliOcrOptions(
                force_full_page_ocr=force,
                lang=langs,
                path=kwargs.get("tesseract_path"),
            )
        elif ocr_engine == "tesseract":
            return TesseractOcrOptions(force_full_page_ocr=force, lang=langs)
        elif ocr_engine == "rapidocr":
            return RapidOcrOptions(force_full_page_ocr=force)
        return EasyOcrOptions(force_full_page_ocr=force, lang=langs)

    @staticmethod
    def _create_table_options(mode: str, do_cell_matching: bool) -> TableStructureOptions:
        """Create table structure options."""
        return TableStructureOptions(
            mode=TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST,
            do_cell_matching=do_cell_matching,
        )

    def _create_pdf_pipeline_options(self, config: Dict[str, Any]) -> PdfPipelineOptions:
        """Create PDF pipeline options for standard processing."""
        ocr_opts = self._create_ocr_options(
            config.get("ocr_engine", "easyocr"),
            force_full_page_ocr=config.get("force_full_page_ocr", False),
            languages=config.get("ocr_languages", ["en"]),
        )
        table_opts = self._create_table_options(
            config.get("table_mode", "accurate"),
            config.get("do_cell_matching", True),
        )
        return PdfPipelineOptions(
            do_ocr=config.get("do_ocr", True),
            do_table_structure=config.get("do_table_structure", True),
            do_code_enrichment=config.get("do_code_enrichment", False),
            do_formula_enrichment=config.get("do_formula_enrichment", False),
            do_picture_classification=config.get("do_picture_classification", False),
            do_picture_description=config.get("do_picture_description", False),
            generate_page_images=config.get("generate_page_images", False),
            generate_picture_images=config.get("generate_picture_images", False),
            images_scale=config.get("images_scale", 1.0),
            backend=getattr(PdfBackend, config.get("pdf_backend", "DLPARSE_V4")),
            accelerator_options=AcceleratorOptions(
                num_threads=min(8, os.cpu_count() or 4),
                device="auto",
            ),
            enable_remote_services=config.get("enable_remote_services", False),
            ocr_options=ocr_opts,
            table_structure_options=table_opts,
        )

    def _create_vlm_pipeline_options(self, config: Dict[str, Any]) -> Tuple[VlmPipelineOptions, type]:
        """Create VLM pipeline options with Ollama support."""
        vlm_model = config.get("vlm_model", "smoldocling")

        pipeline_options = VlmPipelineOptions(
            enable_remote_services=config.get("enable_remote_services", True),
        )

        # Handle different VLM model types
        if vlm_model == "qwen3_vl_8b_ollama":
            # Use custom Ollama URL if provided
            ollama_url = config.get("ollama_url", "http://localhost:11434/v1/chat/completions")
            pipeline_options.vlm_options = ApiVlmOptions(
                url=ollama_url,
                params={"model": config.get("ollama_model_name", "qwen3-vl:8b")},
                prompt=config.get("vlm_prompt", "Convert this page to markdown. Do not miss any text and only output the bare markdown!"),
                scale=config.get("vlm_scale", 2.0),
                timeout=config.get("vlm_timeout", 120),
                response_format=ResponseFormat.MARKDOWN,
                temperature=config.get("vlm_temperature", 0.0),
            )
        elif vlm_model == "granite_vision_ollama":
            ollama_url = config.get("ollama_url", "http://localhost:11434/v1/chat/completions")
            pipeline_options.vlm_options = ApiVlmOptions(
                url=ollama_url,
                params={"model": config.get("ollama_model_name", "granite3.2-vision:2b")},
                prompt=config.get("vlm_prompt", "Convert this page to markdown. Do not miss any text and only output the bare markdown!"),
                scale=config.get("vlm_scale", 1.0),
                timeout=config.get("vlm_timeout", 120),
                response_format=ResponseFormat.MARKDOWN,
                temperature=config.get("vlm_temperature", 0.0),
            )
        elif vlm_model == "custom_ollama":
            # Fully custom Ollama configuration
            ollama_url = config.get("ollama_url", "http://localhost:11434/v1/chat/completions")
            pipeline_options.vlm_options = ApiVlmOptions(
                url=ollama_url,
                params={"model": config.get("ollama_model_name", "qwen3-vl:8b")},
                prompt=config.get("vlm_prompt", "Convert this page to markdown. Do not miss any text and only output the bare markdown!"),
                scale=config.get("vlm_scale", 2.0),
                timeout=config.get("vlm_timeout", 120),
                response_format=ResponseFormat.MARKDOWN,
                temperature=config.get("vlm_temperature", 0.0),
            )
        elif vlm_model == "granite_vision":
            pipeline_options.vlm_options = GRANITE_VISION_TRANSFORMERS
        elif vlm_model == "smoldocling":
            pipeline_options.vlm_options = SMOLDOCLING_TRANSFORMERS
            # Use MLX on macOS if available
            if sys.platform == "darwin":
                try:
                    import mlx_vlm
                    pipeline_options.vlm_options = SMOLDOCLING_MLX
                except ImportError:
                    pass
        else:
            pipeline_options.vlm_options = SMOLDOCLING_TRANSFORMERS

        return pipeline_options, VlmPipeline

    def initialize_converter(self, **config) -> bool:
        """Initialize the Docling converter with given configuration."""
        try:
            pipeline_type = config.get("pipeline_type", "standard")

            if pipeline_type == "vlm":
                pipeline_options, pipeline_cls = self._create_vlm_pipeline_options(config)
                pdf_format_option = PdfFormatOption(
                    pipeline_cls=pipeline_cls,
                    pipeline_options=pipeline_options,
                )
            else:
                pipeline_options = self._create_pdf_pipeline_options(config)
                pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

            format_options = {
                InputFormat.PDF: pdf_format_option,
                InputFormat.IMAGE: pdf_format_option,
            }

            self.converter = get_document_converter(format_options)
            st.success("Docling converter initialized successfully")
            return True

        except Exception as e:
            st.error(f"Error initializing converter: {e}")
            st.code(str(e))
            return False

    def process_from_stream(
        self,
        file_content: bytes,
        filename: str,
        export_format: str = "markdown",
        **options,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a document from binary stream."""
        if not self.converter:
            raise RuntimeError("Converter not initialized")

        try:
            stream = BytesIO(file_content)
            stream.seek(0)
            doc_stream = DocumentStream(name=filename, stream=stream)

            with st.spinner(f"Processing: {filename}"):
                result = self.converter.convert(doc_stream)

            # Extract content based on format
            if export_format == "markdown":
                content = result.document.export_to_markdown()
            elif export_format == "html":
                content = result.document.export_to_html()
            elif export_format == "json":
                content = result.document.export_to_json()
            else:
                content = result.document.export_to_markdown()

            # Build metadata
            metadata = {
                "source_file": filename,
                "file_size": len(file_content),
                "export_format": export_format,
                "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "docling_version": getattr(docling, "__version__", "unknown"),
                "page_count": len(result.document.pages) if hasattr(result.document, "pages") else 0,
                "conversion_status": "success",
            }

            return content, metadata

        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
            raise

    @staticmethod
    def cleanup_resources():
        """Clean up memory resources."""
        if TORCH_AVAILABLE and torch and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def create_chunks(
        content: str, config: Dict[str, Any], tokenizer=None
    ) -> List[Dict[str, Any]]:
        """Create document chunks for RAG applications."""
        max_chars = config.get("max_chunk_size", 2000)
        overlap = config.get("overlap", 200)

        if tokenizer:
            tokens = tokenizer.encode(content, add_special_tokens=False)
            chunks = []
            pos = 0
            while pos < len(tokens):
                end = min(pos + max_chars, len(tokens))
                chunk = tokenizer.decode(tokens[pos:end])
                chunks.append(chunk)
                pos = end - overlap
        else:
            chunks = [
                content[i : i + max_chars]
                for i in range(0, len(content), max_chars - overlap)
            ]

        return [
            {
                "content": c,
                "metadata": {"chunk_index": i + 1, "total_chunks": len(chunks)},
            }
            for i, c in enumerate(chunks)
        ]

    @staticmethod
    def create_zip_package(
        main_name: str,
        main_content: str,
        metadata: Dict[str, Any],
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> bytes:
        """Create a ZIP package with all outputs."""
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            ext = main_name.split(".")[-1]
            base = ".".join(main_name.split(".")[:-1])
            main_ext = ext if ext in {"html", "json"} else "md"
            main_file = f"{base}.{main_ext}"

            # Add frontmatter for markdown
            if main_ext == "md":
                fm = "---\n" + "\n".join(
                    f"{k}: {json.dumps(v)}" for k, v in metadata.items()
                ) + "\n---\n\n"
                full = fm + main_content
            else:
                full = main_content

            zf.writestr(main_file, full)
            zf.writestr(f"{base}_metadata.json", json.dumps(metadata, indent=2))

            if chunks:
                for ch in chunks:
                    idx = ch["metadata"]["chunk_index"]
                    fname = f"{base}_chunks/chunk_{idx:03d}.{main_ext}"
                    chunk_content = ch["content"]
                    if main_ext == "md":
                        cm = "---\n" + "\n".join(
                            f"{k}: {json.dumps(v)}" for k, v in ch["metadata"].items()
                        ) + "\n---\n\n"
                        chunk_content = cm + chunk_content
                    zf.writestr(fname, chunk_content)

        buffer.seek(0)
        return buffer.getvalue()


def main():
    """Main Streamlit application."""
    if not DOCLING_AVAILABLE:
        st.stop()

    # Display version
    try:
        st.success(f"Docling version: {docling.__version__}")
    except Exception:
        st.info("Docling version: unknown")

    # Initialize session state
    if "processor" not in st.session_state:
        st.session_state.processor = DoclingProcessor()
    if "processed_content" not in st.session_state:
        st.session_state.processed_content = None
    if "processed_metadata" not in st.session_state:
        st.session_state.processed_metadata = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Pipeline selection
        pipeline_type = st.radio(
            "Processing Pipeline:",
            ["standard", "vlm"],
            index=0,
            help="Standard = OCR + ML models, VLM = Vision-Language Models (including Ollama)",
        )

        # Export format
        export_format = st.selectbox(
            "Export Format:",
            ["markdown", "html", "json"],
            index=0,
        )

        config: Dict[str, Any] = {
            "pipeline_type": pipeline_type,
        }

        st.divider()

        if pipeline_type == "standard":
            st.subheader("Standard Pipeline Settings")

            pdf_backend = st.selectbox(
                "PDF Backend:",
                ["DLPARSE_V4", "DLPARSE_V2", "DLPARSE_V1", "PYPDFIUM2"],
                help="DLPARSE_V4: Latest, PYPDFIUM2: Most forgiving",
            )
            config["pdf_backend"] = pdf_backend

            do_ocr = st.checkbox("Enable OCR", value=True)
            config["do_ocr"] = do_ocr

            if do_ocr:
                ocr_engine = st.selectbox(
                    "OCR Engine:",
                    ["easyocr", "tesseract_cli", "tesseract", "rapidocr"],
                )
                config["ocr_engine"] = ocr_engine

                force_full_page_ocr = st.checkbox("Force Full Page OCR", value=False)
                config["force_full_page_ocr"] = force_full_page_ocr

                ocr_languages = st.multiselect(
                    "OCR Languages:",
                    ["en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko"],
                    default=["en"],
                )
                config["ocr_languages"] = ocr_languages

            do_table_structure = st.checkbox("Enable Table Structure", value=True)
            config["do_table_structure"] = do_table_structure

            if do_table_structure:
                table_mode = st.selectbox("Table Mode:", ["accurate", "fast"])
                config["table_mode"] = table_mode
                do_cell_matching = st.checkbox("Cell Matching", value=True)
                config["do_cell_matching"] = do_cell_matching

            do_code_enrichment = st.checkbox("Code Detection", value=False)
            config["do_code_enrichment"] = do_code_enrichment

            do_formula_enrichment = st.checkbox("Formula Detection", value=False)
            config["do_formula_enrichment"] = do_formula_enrichment

        else:  # VLM pipeline
            st.subheader("VLM Pipeline Settings")

            vlm_model = st.selectbox(
                "VLM Model:",
                [
                    "qwen3_vl_8b_ollama",
                    "granite_vision_ollama",
                    "custom_ollama",
                    "smoldocling",
                    "granite_vision",
                ],
                index=0,
                help="Ollama models require Ollama running locally",
            )
            config["vlm_model"] = vlm_model

            # Show Ollama configuration for API models
            if vlm_model in ["qwen3_vl_8b_ollama", "granite_vision_ollama", "custom_ollama"]:
                st.info("Ensure Ollama is running: `ollama run qwen3-vl:8b`")

                ollama_url = st.text_input(
                    "Ollama API URL:",
                    value="http://localhost:11434/v1/chat/completions",
                )
                config["ollama_url"] = ollama_url

                if vlm_model == "custom_ollama":
                    ollama_model_name = st.text_input(
                        "Model Name:",
                        value="qwen3-vl:8b",
                        help="Model name as registered in Ollama",
                    )
                    config["ollama_model_name"] = ollama_model_name

                vlm_timeout = st.slider("Timeout (seconds)", 60, 600, 300)
                config["vlm_timeout"] = vlm_timeout

                config["enable_remote_services"] = True
            else:
                config["enable_remote_services"] = False

            vlm_prompt = st.text_area(
                "VLM Prompt:",
                value="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
                help="Instructions for the vision model",
            )
            config["vlm_prompt"] = vlm_prompt

            vlm_scale = st.slider("Image Scale", 0.5, 3.0, 2.0, 0.1)
            config["vlm_scale"] = vlm_scale

            vlm_temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
            config["vlm_temperature"] = vlm_temperature

        st.divider()
        st.subheader("Chunking")

        enable_chunking = st.checkbox("Enable Chunking", value=False)

        if enable_chunking:
            max_chunk_size = st.slider("Max Chunk Size (chars)", 500, 5000, 2000)
            overlap = st.slider("Overlap (chars)", 0, 500, 200)
        else:
            max_chunk_size = 2000
            overlap = 200

        st.session_state.chunking_config = {
            "max_chunk_size": max_chunk_size,
            "overlap": overlap,
        }

    # Main content area
    st.header("Document Processing")

    uploaded_files = st.file_uploader(
        "Upload documents:",
        type=st.session_state.processor.supported_formats,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, PPTX, XLSX, HTML, Images, Markdown, AsciiDoc",
    )

    if not uploaded_files:
        st.info("Drag and drop files or click to upload")

    if uploaded_files and st.button("Process Documents", type="primary"):
        if not st.session_state.processor.initialize_converter(**config):
            st.stop()

        file_obj = uploaded_files[0]
        try:
            content, metadata = st.session_state.processor.process_from_stream(
                file_obj.getvalue(),
                file_obj.name,
                export_format,
            )
            st.session_state.processed_content = content
            st.session_state.processed_metadata = metadata
            st.session_state.chunks = None

            st.success(f"Processed: {file_obj.name}")

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{len(file_obj.getvalue())/1024:.1f} KB")
            with col2:
                st.metric("Output Size", f"{len(content)/1024:.1f} KB")
            with col3:
                st.metric("Pages", metadata.get("page_count", "N/A"))

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            st.session_state.processor.cleanup_resources()

    # Display results
    if st.session_state.processed_content and st.session_state.processed_metadata:
        st.divider()

        # Metadata editor
        st.subheader("Document Metadata")
        meta_df = pd.DataFrame(
            list(st.session_state.processed_metadata.items()),
            columns=["Field", "Value"],
        )
        edited_df = st.data_editor(
            meta_df,
            use_container_width=True,
            num_rows="dynamic",
            key="metadata_editor",
        )

        if st.button("Update Metadata"):
            new_meta = dict(zip(edited_df["Field"], edited_df["Value"]))
            st.session_state.processed_metadata = new_meta
            st.success("Metadata updated!")
            st.rerun()

        # Preview
        st.subheader(f"{export_format.title()} Preview")
        preview_height = st.slider("Preview Height", 200, 800, 400)

        with st.container(height=preview_height):
            if export_format == "markdown":
                st.markdown(st.session_state.processed_content)
            elif export_format == "html":
                st.components.v1.html(
                    st.session_state.processed_content,
                    height=preview_height - 60,
                    scrolling=True,
                )
            else:
                st.json(st.session_state.processed_content)

        # Downloads
        col1, col2 = st.columns(2)

        with col1:
            file_base = Path(
                st.session_state.processed_metadata.get("source_file", "document")
            ).stem
            ext = export_format if export_format in {"html", "json"} else "md"
            file_name = f"{file_base}.{ext}"

            if export_format == "markdown":
                download_payload = (
                    "---\n"
                    + "\n".join(
                        f"{k}: {json.dumps(v)}"
                        for k, v in st.session_state.processed_metadata.items()
                    )
                    + "\n---\n\n"
                    + st.session_state.processed_content
                )
            else:
                download_payload = st.session_state.processed_content

            st.download_button(
                label=f"Download {export_format.upper()}",
                data=download_payload,
                file_name=file_name,
                mime={
                    "markdown": "text/markdown",
                    "html": "text/html",
                    "json": "application/json",
                }[export_format],
            )

        with col2:
            if enable_chunking and st.button("Create Chunks"):
                chunks = DoclingProcessor.create_chunks(
                    st.session_state.processed_content,
                    st.session_state.chunking_config,
                )
                st.session_state.chunks = chunks
                st.success(f"Created {len(chunks)} chunks!")
                st.rerun()

        # Chunks display
        if st.session_state.chunks:
            st.divider()
            st.subheader("Document Chunks")

            chunk_sizes = [len(c["content"]) for c in st.session_state.chunks]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", len(st.session_state.chunks))
            with col2:
                st.metric("Avg Size", f"{sum(chunk_sizes) // len(chunk_sizes):,} chars")
            with col3:
                st.metric("Range", f"{min(chunk_sizes):,} - {max(chunk_sizes):,}")

            selected = st.selectbox(
                "Preview chunk:",
                range(len(st.session_state.chunks)),
                format_func=lambda x: f"Chunk {x+1} ({len(st.session_state.chunks[x]['content']):,} chars)",
            )

            ch = st.session_state.chunks[selected]
            with st.expander(f"Chunk {selected + 1} Preview", expanded=True):
                if export_format == "markdown":
                    st.markdown(ch["content"])
                else:
                    st.text(ch["content"])

            # ZIP download
            zip_bytes = DoclingProcessor.create_zip_package(
                f"{file_base}.{ext}",
                st.session_state.processed_content,
                st.session_state.processed_metadata,
                st.session_state.chunks,
            )
            st.download_button(
                label="Download Complete ZIP",
                data=zip_bytes,
                file_name=f"{file_base}_complete.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
