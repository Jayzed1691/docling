#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docling Document Converter – Streamlit UI
────────────────────────────────────────────────────────────────────
Fast, memory‑friendly, fully‑featured.
"""

# ─────── 1. Imports – everything explicitly listed ────────
import os
import re
import json
import sys
import logging
import gc
import warnings
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from io import BytesIO

import streamlit as st
import pandas as pd

# Optional torch (used only for GPU‑aware OCR)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None   

# Docling – optional import; errors are surfaced in the UI
try:
    import docling
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat, DocumentStream
    from docling.datamodel.pipeline_options import (
        PdfBackend,
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
        EasyOcrOptions,
        TesseractCliOcrOptions,
        TesseractOcrOptions,
        RapidOcrOptions,
        VlmPipelineOptions,
        PictureDescriptionVlmOptions,
        AcceleratorOptions,
    )
    from docling.chunking import HybridChunker
    DOCLING_AVAILABLE = True
except Exception as exc:
    DOCLING_AVAILABLE = False
    st.error(f"❌ Docling not installed: {exc}")
    st.info("Install with: `pip install docling`")

# Silence noisy Torch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("torch._C").setLevel(logging.ERROR)

# ───── 2. Cached Docling converter (per‑session) ─────
@st.cache_resource(show_spinner=False)
def get_document_converter(
    _format_options: Dict[InputFormat, PdfFormatOption]
) -> DocumentConverter:
    """Return a shared DocumentConverter instance."""
    return DocumentConverter(format_options=_format_options)

# ───── 3. Helper – preset overrides ─────
def apply_preset(preset: str, pipeline_type: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Override config keys according to the selected preset."""
    if preset == "Large Documents" and pipeline_type == "standard":
        cfg.update(
            {
                "pdf_backend": "DLPARSE_V4",
                "do_picture_classification": False,
                "do_picture_description": False,
                "generate_page_images": False,
                "generate_picture_images": False,
                "images_scale": 1.0,
            }
        )
    elif preset == "Table Optimized" and pipeline_type == "standard":
        cfg.update(
            {
                "table_mode": "accurate",
                "do_cell_matching": True,
                "do_table_structure": True,
            }
        )
    elif preset == "Performance" and pipeline_type == "standard":
        cfg.update(
            {
                "pdf_backend": "DLPARSE_V4",
                "do_ocr": True,
                "do_table_structure": True,
                "do_code_enrichment": False,
                "do_formula_enrichment": False,
                "do_picture_classification": False,
                "do_picture_description": False,
            }
        )
    return cfg

# ───── 4. Processor class – lightweight, reusable ─────
class DoclingProcessor:
    def __init__(self):
        self.converter: Optional[DocumentConverter] = None
        self.supported_formats = [
            "pdf", "docx", "pptx", "xlsx", "html",
            "png", "jpg", "jpeg", "bmp", "tiff",
            "md", "asciidoc",
        ]
        self._lenient_cache: Dict[str, DocumentConverter] = {}

    # ------------------------------------------------------------
    @staticmethod
    def _create_ocr_options(ocr_engine: str, **kwargs) -> Any:
        """Return OCR options; uses GPU when available."""
        force = kwargs.get("force_full_page_ocr", False)
        langs = kwargs.get("languages", ["en"])
        if ocr_engine == "easyocr":
            ocr = EasyOcrOptions(force_full_page_ocr=force, lang=langs)
            if TORCH_AVAILABLE and hasattr(ocr, "use_gpu"):
                ocr.use_gpu = torch.cuda.is_available()
            return ocr
        if ocr_engine == "tesseract_cli":
            return TesseractCliOcrOptions(
                force_full_page_ocr=force,
                lang=langs,
                path=kwargs.get("tesseract_path"),
            )
        if ocr_engine == "tesseract":
            return TesseractOcrOptions(force_full_page_ocr=force, lang=langs)
        if ocr_engine == "rapidocr":
            return RapidOcrOptions(force_full_page_ocr=force)
        # fallback
        return EasyOcrOptions(force_full_page_ocr=force, lang=langs)

    @staticmethod
    def _create_table_options(mode: str, do_cell_matching: bool) -> TableStructureOptions:
        return TableStructureOptions(
            mode=TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST,
            do_cell_matching=do_cell_matching,
        )

    @staticmethod
    def _create_pdf_pipeline_options(config: Dict[str, Any]) -> PdfPipelineOptions:
        """Fast baseline options for PDF pipeline."""
        ocr_opts = DoclingProcessor._create_ocr_options(
            config.get("ocr_engine", "easyocr"),
            force_full_page_ocr=config.get("force_full_page_ocr", False),
            languages=config.get("ocr_languages", ["en"]),
        )
        table_opts = DoclingProcessor._create_table_options(
            config.get("table_mode", "accurate"),
            config.get("do_cell_matching", True),
        )
        return PdfPipelineOptions(
            do_ocr=config.get("do_ocr", True),
            do_table_structure=True,
            do_code_enrichment=config.get("do_code_enrichment", False),
            do_formula_enrichment=config.get("do_formula_enrichment", False),
            do_picture_classification=False,
            do_picture_description=False,
            generate_page_images=False,
            generate_picture_images=False,
            images_scale=config.get("images_scale", 1.0),
            backend=getattr(PdfBackend, config.get("pdf_backend", "DLPARSE_V4")),
            accelerator_options=AcceleratorOptions(
                num_threads=min(8, os.cpu_count()),
                device="auto",
            ),
            enable_remote_services=config.get("enable_remote_services", False),
            ocr_options=ocr_opts,
            table_structure_options=table_opts,
        )

    # ------------------------------------------------------------
    def initialize_converter(self, **config) -> bool:
        """Instantiate (cached) converter according to the configuration."""
        try:
            pipeline_opt = self._create_pdf_pipeline_options(config)
            fmt_opt = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opt)}
            self.converter = get_document_converter(fmt_opt)
            st.success("✅ Docling converter ready")
            return True
        except Exception as exc:
            st.error(f"❌ Cannot create converter: {exc}")
            st.code(str(exc))
            return False

    # ------------------------------------------------------------
    def get_lenient_converter(self, backend: str) -> DocumentConverter:
        """Return a converter that uses the PYPDFIUM2 backend – shared per backend."""
        if backend not in self._lenient_cache:
            opts = PdfPipelineOptions(
                backend=getattr(PdfBackend, backend),
                do_ocr=True,
                do_table_structure=False,
                generate_page_images=False,
                images_scale=1.0,
            )
            fmt_opt = {InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            self._lenient_cache[backend] = get_document_converter(fmt_opt)
        return self._lenient_cache[backend]

    # ------------------------------------------------------------
    def process_document(
        self, file_path: str, export_format: str = "markdown", **options
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.converter:
            raise RuntimeError("Converter not initialized – call initialize_converter() first")
        try:
            with st.spinner(f"🔄 Processing: {Path(file_path).name}"):
                result = self.converter.convert(file_path, **options)
            content = self._extract_output(result, export_format)
            metadata = self._build_metadata(file_path, result.document, export_format)
            return content, metadata
        except Exception as exc:
            st.error(f"❌ Error processing file: {exc}")
            raise

    def process_from_stream(
        self,
        file_content: bytes,
        filename: str,
        export_format: str = "markdown",
        **options,
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.converter:
            raise RuntimeError("Converter not initialized – call initialize_converter() first")
        try:
            stream = BytesIO(file_content)
            stream.seek(0)
            doc_stream = DocumentStream(name=filename, stream=stream)
            with st.spinner(f"🔄 Processing: {filename}"):
                result = self.converter.convert(doc_stream, **options)
            content = self._extract_output(result, export_format)
            metadata = self._build_metadata(filename, result.document, export_format)
            return content, metadata
        except Exception as exc:
            st.error(f"❌ Error processing {filename}: {exc}")
            raise

    # ------------------------------------------------------------
    @staticmethod
    def _extract_output(result, export_format: str) -> str:
        if export_format == "markdown":
            return result.document.export_to_markdown()
        if export_format == "html":
            return result.document.export_to_html()
        if export_format == "json":
            return result.document.export_to_json()
        return result.document.export_to_markdown()

    @staticmethod
    def _build_metadata(filename, document, export_format: str) -> Dict[str, Any]:
        # if file is on disk, get real size; otherwise skip in-memory size
        file_size = os.path.getsize(filename) if os.path.exists(filename) else None
        return {
            "source_file": Path(filename).name,
            "file_size": file_size,
            "export_format": export_format,
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "docling_version": getattr(docling, "__version__", "latest"),
            "conversion_status": "success",
        }
        if hasattr(document, "metadata") and document.metadata:
            meta.update(document.metadata)
        return meta

    # ------------------------------------------------------------
    @staticmethod
    def _cleanup_resources():
        """Remove intermediate objects and free CUDA memory."""
        if TORCH_AVAILABLE and torch and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    @staticmethod
    def _build_chunks(text: str, cfg: Dict[str, Any], tokenizer=None) -> List[Dict]:
        max_chars = cfg.get("max_chunk_size", 2000)
        overlap = cfg.get("overlap", 200)

        if tokenizer:
            # Token‑aware (works for HuggingFace tokenizers)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            chunks = []
            pos = 0
            while pos < len(tokens):
                end = min(pos + max_chars, len(tokens))
                chunk = tokenizer.decode(tokens[pos:end])
                chunks.append(chunk)
                pos = end - overlap
        else:
            # plain split
            chunks = [
                text[i : i + max_chars] for i in range(0, len(text), max_chars - overlap)
            ]

        return [
            {
                "content": c,
                "metadata": {"chunk_index": i + 1, "total_chunks": len(chunks)},
            }
            for i, c in enumerate(chunks)
        ]

    # ------------------------------------------------------------
    @staticmethod
    def _stream_zip(
        main_name: str,
        main_content: str,
        metadata: Dict,
        chunks: Optional[List[Dict]] = None,
    ) -> bytes:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            ext = main_name.split(".")[-1]
            base = ".".join(main_name.split(".")[:-1])
            main_ext = ext if ext in {"html", "json"} else "md"
            main_file = f"{base}.{main_ext}"
            if main_ext == "md":
                fm = "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in metadata.items()) + "\n---\n\n"
                full = fm + main_content
            else:
                full = main_content
            zf.writestr(main_file, full)
            zf.writestr(f"{base}_metadata.json", json.dumps(metadata, indent=2))
            if chunks:
                for ch in chunks:
                    idx = ch["metadata"]["chunk_index"]
                    fname = f"{base}_chunks/chunk_{idx:03d}.{main_ext}"
                    content = ch["content"]
                    if main_ext == "md":
                        cm = "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in ch["metadata"].items()) + "\n---\n\n"
                        content = cm + content
                    zf.writestr(fname, content)
        buffer.seek(0)
        return buffer.getvalue()

# ───── 5. Streamlit UI ─────
PAGE_MARGINS = 20

st.set_page_config(
    page_title="Docling Document Converter",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not DOCLING_AVAILABLE:
    st.stop()

# Show version
try:
    st.success(f"📦 Docling version: {docling.__version__}")
except Exception:
    st.info("📦 Docling version: cannot detect")

# Session state
if "processor" not in st.session_state:
    st.session_state.processor = DoclingProcessor()
if "processed_content" not in st.session_state:
    st.session_state.processed_content = None
if "processed_metadata" not in st.session_state:
    st.session_state.processed_metadata = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "debug_show" not in st.session_state:
    st.session_state.debug_show = False

# Sidebar – configuration
with st.sidebar:
    st.header("⚙️ Docling Configuration")

    pipeline_type = st.radio(
        "Processing Pipeline:",
        ["standard", "vlm"],
        index=0,
        help="Standard = OCR + ML, VLM = Vision‑Language Models",
    )

    preset_choice = st.selectbox(
        "⚡ Quick Presets",
        ["Custom", "Large Documents", "Table Optimized", "Performance"],
        index=0,
        help="Pre‑defined settings for common use cases",
    )

    config: Dict[str, Any] = {}

    export_format = st.selectbox("📄 Export Format:", ["markdown", "html", "json"])

    if pipeline_type == "standard":
        pdf_backend = st.selectbox(
            "PDF Backend:",
            ["DLPARSE_V2", "DLPARSE_V4", "DLPARSE_V1", "PYPDFIUM2"],
            help="Fastest: DLPARSE_V4; most forgiving: PYPDFIUM2",
        )
        config.update({"pdf_backend": pdf_backend})

        do_ocr = st.checkbox("Enable OCR", value=True)
        config.update({"do_ocr": do_ocr})
        if do_ocr:
            ocr_engine = st.selectbox(
                "OCR Engine:",
                ["easyocr", "tesseract_cli", "tesseract", "rapidocr"],
                help="Choose OCR backend",
            )
            config.update({"ocr_engine": ocr_engine})
            force_full_page_ocr = st.checkbox("Force Full Page OCR", value=False)
            config.update({"force_full_page_ocr": force_full_page_ocr})
            ocr_languages = st.multiselect(
                "OCR Languages:",
                ["en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko"],
                default=["en"],
            )
            config.update({"ocr_languages": ocr_languages})

        do_table_structure = st.checkbox("Enable Table Structure", value=True)
        config.update({"do_table_structure": do_table_structure})
        if do_table_structure:
            table_mode = st.selectbox(
                "Table Mode:", ["accurate", "fast"], help="Accurate = higher quality"
            )
            config.update({"table_mode": table_mode})
            do_cell_matching = st.checkbox("Enable Cell Matching", value=True)
            config.update({"do_cell_matching": do_cell_matching})
        else:
            config.update({"table_mode": "accurate", "do_cell_matching": True})

        do_code_enrichment = st.checkbox("Code Detection", value=False)
        config.update({"do_code_enrichment": do_code_enrichment})
        do_formula_enrichment = st.checkbox("Formula Detection", value=False)
        config.update({"do_formula_enrichment": do_formula_enrichment})
        do_picture_classification = st.checkbox("Picture Classification", value=False)
        config.update({"do_picture_classification": do_picture_classification})
        do_picture_description = st.checkbox("Picture Description", value=False)
        config.update({"do_picture_description": do_picture_description})

        generate_page_images = st.checkbox("Generate Page Images", value=False)
        config.update({"generate_page_images": generate_page_images})
        generate_picture_images = st.checkbox("Generate Picture Images", value=False)
        config.update({"generate_picture_images": generate_picture_images})
        images_scale = st.slider("Images Scale", 0.5, 2.0, 1.0, 0.1)
        config.update({"images_scale": images_scale})

    else:  # VLM
        vlm_model = st.selectbox("VLM Model:", ["smoldocling", "granite_vision"])
        config.update({"vlm_model": vlm_model})

        vlm_prompt = st.text_area(
            "VLM Prompt:",
            value="Convert this document to markdown format, preserving structure and content.",
            help="Prompt that the VLM will see",
        )
        config.update({"vlm_prompt": vlm_prompt})

        vlm_scale = st.slider("Image Scale", 0.5, 2.0, 1.0, 0.1)
        config.update({"vlm_scale": vlm_scale})

        device = st.selectbox("Device:", ["auto", "cpu", "cuda", "mps"])
        config.update({"device": device})
        num_threads = st.slider("Number of Threads", 1, 16, 4)
        config.update({"num_threads": num_threads})

    # Apply preset overrides
    config = apply_preset(preset_choice, pipeline_type, config)

    # Limits
    max_pages = st.number_input(
        "Max Pages (0 = unlimited)",
        min_value=0,
        value=0,
    )
    max_file_size_mb = st.number_input(
        "Max File Size (MB, 0 = unlimited)",
        min_value=0,
        value=0,
    )
    config.update(
        {
            "max_pages": max_pages if max_pages > 0 else None,
            "max_file_size": max_file_size_mb * 1024 * 1024 if max_file_size_mb > 0 else None,
        }
    )

    config.update({"enable_remote_services": st.checkbox(
        "🌐 Enable Remote Services",
        value=False
    )})

    # Chunking
    st.divider()
    enable_chunking = st.checkbox("Enable Chunking", value=False)

    if enable_chunking:
        tokenizer = st.selectbox(
            "Tokenizer:",
            ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
        )
        max_tokens = st.slider("Max Tokens per Chunk", 128, 1024, 512)
        max_chunk_size = st.slider("Max Chunk Size (chars)", 500, 5000, 2000)
        overlap = st.slider("Overlap (chars)", 0, 500, 200)
    else:
        tokenizer = None
        max_tokens = None
        max_chunk_size = None
        overlap = None

    st.session_state.chunking_config = {
        "tokenizer": tokenizer,
        "max_tokens": max_tokens,
        "max_chunk_size": max_chunk_size,
        "overlap": overlap,
    }

# Main content – file upload & processing
st.header("📁 Document Processing")

uploaded_files = st.file_uploader(
    "Upload documents:",
    type=st.session_state.processor.supported_formats,
    accept_multiple_files=True,
    help="Supported formats: PDF, DOCX, PPTX, XLSX, HTML, images, Markdown, AsciiDoc",
)

if not uploaded_files:
    st.info("💡 Drag and drop files here or use the button above!")

if uploaded_files and st.button("🚀 Process Documents", type="primary"):
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
        st.success(f"✅ Processed: {file_obj.name}")

        # Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{len(file_obj.getvalue())/1024:.1f} KB")
        with col2:
            st.metric("Output Size", f"{len(content)/1024:.1f} KB")
        with col3:
            st.metric("Pages", metadata.get("page_count", "N/A"))

    except Exception as exc:
        st.error(f"❌ Error processing {file_obj.name}: {exc}")
        st.session_state.processor._cleanup_resources()
    finally:
        st.session_state.processor._cleanup_resources()

# Show results
if st.session_state.processed_content and st.session_state.processed_metadata:
    st.divider()
    st.subheader("📝 Document Metadata")

    meta_df = pd.DataFrame(
        [
            (k, v)
            for k, v in st.session_state.processed_metadata.items()
            if not k.startswith("docling_")
        ],
        columns=["Field", "Value"],
    )
    edited_df = st.data_editor(
        meta_df,
        use_container_width=True,
        num_rows="dynamic",
        key="metadata_editor",
    )

    if st.button("✅ Update Metadata"):
        new_meta = dict(zip(edited_df["Field"], edited_df["Value"]))
        for k, v in st.session_state.processed_metadata.items():
            if k not in new_meta:
                new_meta[k] = v
        st.session_state.processed_metadata = new_meta
        st.success("✅ Metadata updated!")
        st.rerun()

    # Preview
    st.subheader(f"📄 {export_format.title()} Preview")
    preview_height = st.slider("Preview Height", 200, 800, 400)
    with st.container(height=preview_height):
        if export_format == "markdown":
            st.markdown(st.session_state.processed_content)
        elif export_format == "html":
            st.components.v1.html(st.session_state.processed_content, height=preview_height - 60, scrolling=True)
        else:  # json
            st.json(st.session_state.processed_content)

    # Downloads
    col1, col2 = st.columns(2)
    with col1:
        file_base = Path(st.session_state.processed_metadata.get("source_file", "document")).stem
        ext = export_format if export_format in {"html", "json"} else "md"
        file_name = f"{file_base}.{ext}"
        download_payload = (
            f"---\n"
            + "\n".join(f"{k}: {json.dumps(v)}" for k, v in st.session_state.processed_metadata.items())
            + "\n---\n\n"
            + st.session_state.processed_content
            if export_format == "markdown"
            else st.session_state.processed_content
        )
        st.download_button(
            label=f"💾 Download {export_format.upper()}",
            data=download_payload,
            file_name=file_name,
            mime={"markdown": "text/markdown", "html": "text/html", "json": "application/json"}[export_format],
        )

    with col2:
        if enable_chunking and st.button("✂️ Create Chunks", type="secondary"):
            cfg = st.session_state.chunking_config
            tokenizer_obj = None
            if cfg["tokenizer"]:
                try:
                    from transformers import AutoTokenizer

                    tokenizer_obj = AutoTokenizer.from_pretrained(cfg["tokenizer"])
                except Exception:
                    st.warning("Could not load tokenizer – falling back to plain split")
            chunks = DoclingProcessor._build_chunks(
                st.session_state.processed_content,
                cfg,
                tokenizer=tokenizer_obj,
            )
            st.session_state.chunks = chunks
            st.success(f"✅ Created {len(chunks)} chunks!")
            st.rerun()

    if st.session_state.chunks:
        st.divider()
        st.subheader("✂️ Document Chunks")

        chunk_sizes = [len(c["content"]) for c in st.session_state.chunks]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", len(st.session_state.chunks))
        with col2:
            st.metric("Avg Size", f"{sum(chunk_sizes) // len(chunk_sizes):,} chars")
        with col3:
            st.metric("Size Range", f"{min(chunk_sizes):,} – {max(chunk_sizes):,}")

        selected = st.selectbox(
            "Select chunk to preview:",
            range(len(st.session_state.chunks)),
            format_func=lambda x: f"Chunk {x+1} ({len(st.session_state.chunks[x]['content']):,} chars)",
        )
        ch = st.session_state.chunks[selected]
        with st.expander(f"📄 Chunk {selected+1} Preview", expanded=True):
            if export_format == "markdown":
                st.markdown(ch["content"])
            elif export_format == "html":
                st.components.v1.html(ch["content"], height=250, scrolling=True)
            else:
                st.text(ch["content"])

        with st.expander("📊 Chunk Metadata"):
            st.json(ch["metadata"])

        # ZIP package
        zip_bytes = DoclingProcessor._stream_zip(
            f"{file_base}.{ext}",
            st.session_state.processed_content,
            st.session_state.processed_metadata,
            st.session_state.chunks,
        )
        st.download_button(
            label="📦 Download Complete ZIP",
            data=zip_bytes,
            file_name=f"{file_base}_complete.zip",
            mime="application/zip",
        )

# Optional debug panel
st.sidebar.checkbox(
    "Show Debug Panel",
    key="debug_show",
    value=st.session_state.debug_show,
)

if st.session_state.debug_show:
    st.sidebar.subheader("⚙️ Debug Information")
    st.sidebar.json(
        {
            "Docling version": getattr(docling, "__version__", "unknown"),
            "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "GPU available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "Converter ready": st.session_state.processor.converter is not None,
        }
    )
    