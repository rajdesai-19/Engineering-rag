#!/usr/bin/env python3
"""
rag_retrieval.py — Unified Multimodal Document & Diagram RAG System.

A standalone, private, local alternative to NotebookLM:
- Ingests any PDF documents placed in `knowledge_base/`
- Extracts both full text and embedded diagrams, figures, schematics, and charts
- Builds local dense vector embeddings (FastEmbed BGE) stored in persistent ChromaDB
- Supports free local LLMs via Ollama, Google Gemini API, or OpenAI
- Provides an interactive Streamlit UI with source citations and visual diagram rendering
- Can also be run directly from the command line for headless indexing and querying
"""

import os
import sys
import re
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure UTF-8 output across environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Optional environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import fitz  # PyMuPDF
import chromadb
from fastembed import TextEmbedding

try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


# =============================================================================
# 1. CONFIGURATION & PATH SETUP
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
DB_DIR = BASE_DIR / "vector_db"
IMAGES_DIR = BASE_DIR / "extracted_images"
MODELS_DIR = BASE_DIR / "models"

KB_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COLLECTION = "pdf_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Recommended offline HuggingFace GGUF models (quantized for standard PCs & laptops)
POPULAR_HF_MODELS = {
    "Llama-3.2-3B-Instruct (Q4_K_M, ~2.0 GB)": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "description": "Fast, high-quality 3B model from Meta. Perfect for 8GB+ RAM."
    },
    "Qwen2.5-3B-Instruct (Q4_K_M, ~2.1 GB)": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "description": "Excellent multilingual and technical reasoning 3B model."
    },
    "Qwen2.5-7B-Instruct (Q4_K_M, ~4.7 GB)": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "description": "High-accuracy 7B model. Recommended for 16GB+ RAM."
    },
    "Mistral-7B-Instruct-v0.3 (Q4_K_M, ~4.4 GB)": {
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "description": "Strong general and technical instruction-following 7B model."
    }
}

# Filtering thresholds for diagrams
MIN_IMAGE_WIDTH = 120
MIN_IMAGE_HEIGHT = 120
MAX_IMAGE_ASPECT_RATIO = 12.0  # Filter out decorative horizontal/vertical divider lines


# =============================================================================
# 2. PDF & DIAGRAM EXTRACTION ENGINE
# =============================================================================

class DiagramExtractor:
    """Extracts figures (raster, vector, or composite) and associates their captions."""

    # Anchored at the start of a text block and requiring "." or ":" after the number,
    # so in-text references like "Figure 3 shows ..." are not mistaken for captions
    CAPTION_REGEX = re.compile(
        r"^\s*(Figure|Fig\.?|Diagram|Chart|Schematic)\s*(\d+[A-Za-z]?)\s*[.:]",
        re.IGNORECASE
    )
    MAX_CAPTION_GAP = 80   # max points between a figure's edge and its caption
    MIN_REGION_SIZE = 40   # points; smaller drawing clusters are rules, logos, bullets
    RENDER_DPI = 200

    @classmethod
    def _find_captions(cls, page: fitz.Page) -> List[Tuple[fitz.Rect, str, str]]:
        """Returns (rect, figure_number, caption_text) for each caption block on the page."""
        captions = []
        for block in page.get_text("blocks"):
            if block[6] != 0:  # not a text block
                continue
            m = cls.CAPTION_REGEX.match(block[4])
            if m:
                text = re.sub(r"\s+", " ", block[4]).strip()
                captions.append((fitz.Rect(block[:4]), m.group(2), text[:300]))
        return captions

    @classmethod
    def _figure_regions(cls, page: fitz.Page) -> List[fitz.Rect]:
        """Bounding boxes of vector drawing clusters and placed raster images."""
        regions = []
        try:
            regions.extend(page.cluster_drawings())
        except Exception:
            pass
        for img_info in page.get_images(full=True):
            try:
                regions.extend(page.get_image_rects(img_info[0]))
            except Exception:
                pass
        page_area = page.rect.get_area()
        return [
            r for r in regions
            if r.width >= cls.MIN_REGION_SIZE and r.height >= cls.MIN_REGION_SIZE
            and r.get_area() < 0.8 * page_area  # skip page borders / background frames
        ]

    @classmethod
    def _locate_figure(cls, cap_rect: fitz.Rect, regions: List[fitz.Rect]) -> Optional[fitz.Rect]:
        """Finds the figure area directly above (preferred) or below a caption."""
        def overlaps_horizontally(r):
            return min(r.x1, cap_rect.x1) > max(r.x0, cap_rect.x0)

        above = [(cap_rect.y0 - r.y1, r) for r in regions
                 if overlaps_horizontally(r) and -5 <= cap_rect.y0 - r.y1 <= cls.MAX_CAPTION_GAP]
        below = [(r.y0 - cap_rect.y1, r) for r in regions
                 if overlaps_horizontally(r) and -5 <= r.y0 - cap_rect.y1 <= cls.MAX_CAPTION_GAP]
        candidates = above or below
        if not candidates:
            return None

        figure = fitz.Rect(min(candidates, key=lambda c: c[0])[1])
        # Grow to include overlapping pieces (e.g. raster tiles inside a vector figure)
        grown = True
        while grown:
            grown = False
            for r in regions:
                if figure.intersects(r) and not figure.contains(r):
                    figure |= r
                    grown = True
        return figure

    @classmethod
    def extract_page_diagrams(
        cls,
        doc: fitz.Document,
        page_idx: int,
        doc_output_dir: Path
    ) -> List[Dict[str, Any]]:
        page = doc[page_idx]
        page_num = page_idx + 1
        captions = cls._find_captions(page)
        regions = cls._figure_regions(page)

        extracted = []
        covered = []

        # 1. Captioned figures: render the whole figure area, which captures vector
        #    drawings and multi-tile composites that raw image extraction would miss
        for cap_rect, fig_num, caption in captions:
            figure = cls._locate_figure(cap_rect, regions)
            if figure is None:
                continue
            clip = (figure | cap_rect) + (-6, -6, 6, 6)
            clip &= page.rect
            try:
                pix = page.get_pixmap(dpi=cls.RENDER_DPI, clip=clip)
            except Exception:
                continue
            img_filename = f"page_{page_num:03d}_figure_{re.sub(r'[^A-Za-z0-9]', '', fig_num)}.png"
            img_path = doc_output_dir / img_filename
            pix.save(str(img_path))
            covered.append(clip)
            extracted.append({
                "image_path": str(img_path.relative_to(BASE_DIR)),
                "full_path": str(img_path),
                "page": page_num,
                "width": pix.width,
                "height": pix.height,
                "caption": caption,
                "xref": -1
            })

        # 2. Remaining raster images not part of a captioned figure (e.g. uncaptioned photos)
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:  # image is embedded but never drawn on this page
                    continue
                if any(c.intersects(r) for r in rects for c in covered):
                    continue

                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]

                # Filter out tiny icons, logos, or thin divider lines
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue
                aspect = max(width / height, height / width)
                if aspect > MAX_IMAGE_ASPECT_RATIO:
                    continue

                # Use the vertically nearest caption, if any
                if captions:
                    img_rect = rects[0]
                    caption = min(
                        captions,
                        key=lambda c: min(abs(c[0].y0 - img_rect.y1), abs(img_rect.y0 - c[0].y1))
                    )[2]
                else:
                    caption = f"Image on Page {page_num}"

                img_filename = f"page_{page_num:03d}_img_{img_idx + 1:02d}.{image_ext}"
                img_path = doc_output_dir / img_filename
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                extracted.append({
                    "image_path": str(img_path.relative_to(BASE_DIR)),
                    "full_path": str(img_path),
                    "page": page_num,
                    "width": width,
                    "height": height,
                    "caption": caption,
                    "xref": xref
                })
            except Exception:
                continue

        return extracted


class PDFProcessor:
    """Processes PDF files into structured text and diagram chunks."""

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Packs sentences into ~chunk_size character windows with sentence-level overlap.

        chunk_size is kept well under the embedding model's 512-token limit
        (~2000 chars for BGE) so no chunk content is silently truncated.
        """
        if not text or not text.strip():
            return []

        # PyMuPDF separates layout lines with single newlines: re-join hyphenated
        # words and wrapped lines, keeping blank lines as paragraph breaks
        text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
        paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]

        # Break paragraphs into sentence units; hard-split any unit longer than chunk_size
        units = []
        for para in paragraphs:
            if not para:
                continue
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", para):
                while len(sent) > chunk_size:
                    cut = sent.rfind(" ", 0, chunk_size)
                    if cut <= 0:
                        cut = chunk_size
                    units.append(sent[:cut].strip())
                    sent = sent[cut:].strip()
                if sent:
                    units.append(sent)

        chunks = []
        current = []
        current_len = 0

        for unit in units:
            if current and current_len + len(unit) + 1 > chunk_size:
                chunks.append(" ".join(current))
                # Carry trailing sentences forward as overlap (at least one, unless it is huge)
                overlap = []
                overlap_len = 0
                for prev in reversed(current):
                    if overlap_len + len(prev) > chunk_overlap and (overlap or len(prev) > chunk_size // 2):
                        break
                    overlap.insert(0, prev)
                    overlap_len += len(prev) + 1
                # Shrink overlap if it would push the next chunk past chunk_size
                while overlap and overlap_len + len(unit) + 1 > chunk_size:
                    overlap_len -= len(overlap.pop(0)) + 1
                current, current_len = overlap, overlap_len
            current.append(unit)
            current_len += len(unit) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks

    @classmethod
    def process_pdf(cls, pdf_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extracts text chunks and diagram chunks from a single PDF."""
        doc = fitz.open(str(pdf_path))
        doc_name = pdf_path.stem
        doc_img_dir = IMAGES_DIR / doc_name
        doc_img_dir.mkdir(parents=True, exist_ok=True)
        # Clear images from previous runs so stale figures are not left behind
        for old_img in doc_img_dir.iterdir():
            if old_img.is_file():
                old_img.unlink()

        text_chunks = []
        all_diagrams = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Extract diagrams
            page_diagrams = DiagramExtractor.extract_page_diagrams(doc, page_idx, doc_img_dir)
            all_diagrams.extend(page_diagrams)

            # 2. Extract page text
            page_text = page.get_text("text").strip()
            if not page_text and page_diagrams:
                page_text = f"Page {page_num} contains diagrams: " + "; ".join(d["caption"] for d in page_diagrams)

            # 3. Create text chunks
            p_chunks = cls.chunk_text(page_text)
            for c_idx, chunk_str in enumerate(p_chunks):
                meta = {
                    "document": doc_name,
                    "source_file": pdf_path.name,
                    "page": page_num,
                    "chunk_index": c_idx,
                    "has_diagram": len(page_diagrams) > 0,
                    "diagram_path": page_diagrams[0]["image_path"] if page_diagrams else "",
                    "diagram_caption": page_diagrams[0]["caption"] if page_diagrams else "",
                    "is_diagram_chunk": False,
                }
                text_chunks.append({
                    "id": f"{doc_name}_p{page_num}_c{c_idx}",
                    "text": chunk_str,
                    "metadata": meta
                })

            # 4. Create dedicated searchable diagram chunks for direct diagram lookups
            for d_idx, diag in enumerate(page_diagrams):
                diag_text = (
                    f"[FIGURE / DIAGRAM from {doc_name}, Page {page_num}]\n"
                    f"Caption: {diag['caption']}\n"
                    f"Context: {page_text[:400]}"
                )
                text_chunks.append({
                    "id": f"{doc_name}_p{page_num}_diag_{d_idx}",
                    "text": diag_text,
                    "metadata": {
                        "document": doc_name,
                        "source_file": pdf_path.name,
                        "page": page_num,
                        "chunk_index": 9000 + d_idx,
                        "has_diagram": True,
                        "diagram_path": diag["image_path"],
                        "diagram_caption": diag["caption"],
                        "is_diagram_chunk": True,
                    }
                })

        doc.close()
        return text_chunks, all_diagrams


# =============================================================================
# 3. VECTOR DATABASE & SEMANTIC RETRIEVER
# =============================================================================

class VectorDatabase:
    """Manages local ChromaDB indexing with FastEmbed BGE dense embeddings."""

    def __init__(
        self,
        db_path: Path = DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_EMBEDDING_MODEL
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        # Cache inside models/ (not the OS temp dir) so the app keeps working offline
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=str(MODELS_DIR / "fastembed"))

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def _indexed_hashes(self) -> Dict[str, str]:
        """Maps each indexed source_file to the file hash it was indexed from."""
        res = self.collection.get(include=["metadatas"])
        return {
            m["source_file"]: m.get("file_hash", "")
            for m in (res.get("metadatas") or [])
            if m and m.get("source_file")
        }

    def build_or_update(
        self,
        kb_path: Path = KB_DIR,
        progress_callback=None,
        force: bool = False
    ) -> Dict[str, Any]:
        """Incrementally syncs ChromaDB with knowledge_base.

        New or changed PDFs are (re)indexed, unchanged PDFs are skipped unless
        force=True, and chunks/images of PDFs no longer present are removed.
        """
        pdf_files = sorted(kb_path.glob("*.pdf"))
        indexed = self._indexed_hashes()

        # Remove documents that were deleted from the knowledge base
        current_names = {p.name for p in pdf_files}
        removed_files = sorted(set(indexed) - current_names)
        for name in removed_files:
            self.collection.delete(where={"source_file": name})
            stale_img_dir = IMAGES_DIR / Path(name).stem
            if stale_img_dir.is_dir():
                for old_img in stale_img_dir.iterdir():
                    if old_img.is_file():
                        old_img.unlink()
                stale_img_dir.rmdir()

        total_chunks = 0
        total_diagrams = 0
        processed_files = []
        skipped_files = []

        total_files = len(pdf_files)
        for i, pdf_file in enumerate(pdf_files):
            file_hash = self._file_hash(pdf_file)
            if not force and indexed.get(pdf_file.name) == file_hash:
                skipped_files.append(pdf_file.name)
                continue

            if progress_callback:
                progress_callback((i / total_files), f"Processing {pdf_file.name} ({i+1}/{total_files})...")

            chunks, diagrams = PDFProcessor.process_pdf(pdf_file)
            total_diagrams += len(diagrams)

            # Drop the previous version's chunks so none are left orphaned
            self.collection.delete(where={"source_file": pdf_file.name})

            if chunks:
                texts = [c["text"] for c in chunks]
                ids = [c["id"] for c in chunks]
                metadatas = [{**c["metadata"], "file_hash": file_hash} for c in chunks]

                # Generate dense embeddings locally
                embeddings = list(self.model.embed(texts))
                embeddings_list = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]

                # Upsert into ChromaDB in batches
                batch_size = 64
                for b_start in range(0, len(texts), batch_size):
                    b_end = b_start + batch_size
                    self.collection.upsert(
                        ids=ids[b_start:b_end],
                        documents=texts[b_start:b_end],
                        embeddings=embeddings_list[b_start:b_end],
                        metadatas=metadatas[b_start:b_end]
                    )

                total_chunks += len(chunks)
                processed_files.append(pdf_file.name)

        if progress_callback:
            progress_callback(1.0, f"Indexing complete: {total_chunks} chunks indexed!")

        return {
            "status": "success" if pdf_files else "empty",
            "files_processed": processed_files,
            "files_skipped": skipped_files,
            "files_removed": removed_files,
            "total_chunks": total_chunks,
            "total_diagrams": total_diagrams,
            "collection_count": self.collection.count()
        }


class SemanticRetriever:
    """Retrieves top-k relevant document excerpts and associated diagrams."""

    def __init__(self, vector_db: VectorDatabase, enable_reranking: bool = True):
        self.vector_db = vector_db
        # Ranker is created whenever available so reranking can be toggled per query
        self.ranker = Ranker(cache_dir=str(MODELS_DIR / "flashrank")) if HAS_FLASHRANK else None
        self.enable_reranking = enable_reranking and self.ranker is not None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[str] = None,
        rerank: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """rerank overrides enable_reranking for this call (lets a shared retriever serve many sessions)."""
        if not query or not query.strip():
            return []

        coll_count = self.vector_db.count()
        if coll_count == 0:
            return []

        use_reranker = (self.enable_reranking if rerank is None else rerank) and self.ranker is not None
        n_candidates = max(top_k * 3, 15) if use_reranker else top_k
        n_candidates = min(n_candidates, coll_count)

        query_emb = list(self.vector_db.model.embed([query]))[0]
        query_vec = query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)

        kwargs = {
            "query_embeddings": [query_vec],
            "n_results": n_candidates
        }
        if document_filter:
            kwargs["where"] = {"document": document_filter}

        res = self.vector_db.collection.query(**kwargs)
        if not res or not res["documents"] or not res["documents"][0]:
            return []

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        ids = res["ids"][0]

        items = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            # Cosine distance in Chroma: score = 1.0 - dist
            similarity = max(1.0 - float(dist), 0.0)
            items.append({
                "chunk_id": cid,
                "text": text,
                "document": meta.get("document", "Unknown"),
                "source_file": meta.get("source_file", ""),
                "page": meta.get("page", 1),
                "has_diagram": meta.get("has_diagram", False),
                "diagram_path": meta.get("diagram_path", ""),
                "diagram_caption": meta.get("diagram_caption", ""),
                "is_diagram_chunk": meta.get("is_diagram_chunk", False),
                "similarity_score": round(similarity, 4),
            })

        # FlashRank reranking
        if use_reranker and len(items) > 1:
            try:
                passages = [{"id": it["chunk_id"], "text": it["text"]} for it in items]
                rerank_req = RerankRequest(query=query, passages=passages)
                ranked_passages = self.ranker.rerank(rerank_req)
                
                # Map back to items
                id_to_item = {it["chunk_id"]: it for it in items}
                reranked_items = []
                for rp in ranked_passages[:top_k]:
                    matched = id_to_item.get(rp["id"])
                    if matched:
                        matched["rerank_score"] = round(float(rp.get("score", matched["similarity_score"])), 4)
                        reranked_items.append(matched)
                return reranked_items
            except Exception:
                return items[:top_k]

        return items[:top_k]


# =============================================================================
# 4. MULTI-BACKEND LLM GENERATOR & LOCAL GGUF ENGINE
# =============================================================================

class LocalGGUFEngine:
    """Manages offline GGUF models loaded directly from the `models/` directory."""
    _loaded_model_path: Optional[str] = None
    _llm_instance = None

    @classmethod
    def get_available_models(cls) -> List[Path]:
        """Returns all .gguf model files in the models/ directory."""
        return sorted(list(MODELS_DIR.glob("*.gguf")))

    @classmethod
    def get_llm(cls, model_path: Path, n_ctx: int = 4096, n_threads: Optional[int] = None):
        """Loads and caches the local GGUF model instance."""
        if not HAS_LLAMA_CPP:
            raise ImportError(
                "llama-cpp-python is required to run local GGUF models. "
                "Install it using: pip install llama-cpp-python"
            )
        
        threads = n_threads or max((os.cpu_count() or 4) - 2, 2)
        target_path_str = str(model_path.resolve())

        if (
            cls._llm_instance is None
            or cls._loaded_model_path != target_path_str
            or cls._llm_instance.n_ctx() != n_ctx
        ):
            cls._llm_instance = Llama(
                model_path=target_path_str,
                n_ctx=n_ctx,
                n_threads=threads,
                verbose=False
            )
            cls._loaded_model_path = target_path_str
        return cls._llm_instance

    @classmethod
    def download_from_hf(cls, repo_id: str, filename: str) -> Path:
        """Downloads a model directly from Hugging Face Hub to models/."""
        if not HAS_HF_HUB:
            raise ImportError(
                "huggingface-hub is required for direct downloads. "
                "Install it using: pip install huggingface-hub"
            )
        target = MODELS_DIR / filename
        if target.exists():
            return target
        
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        return Path(downloaded)


class LLMGenerator:
    """Universal LLM client supporting local HuggingFace GGUF models, local Ollama, or Google Gemini."""

    SYSTEM_PROMPT = """You are a rigorous technical research assistant.
Answer the user's question using ONLY the provided document excerpts.

Strict Guidelines:
1. Factual Grounding: Base your answer strictly on the excerpts. If the information is not present, clearly state the limitation.
2. In-Text Citations: Explicitly cite sources in your answer using bracketed notation like `[Doc: <document_name>, Page: <page_number>]`.
3. Diagrams & Schematics: When excerpts describe a figure, diagram, or chart, explicitly reference it.
4. Technical Rigor: Preserve formulas, parameter values, and units faithfully.
5. Format: Provide structured, direct markdown without conversational chatter or thinking tokens.
"""

    LOCAL_N_CTX = 8192          # context window for local GGUF and Ollama models
    MAX_ANSWER_TOKENS = 1500
    MAX_CONTEXT_CHARS = 18000   # ~5-6k tokens of technical text; leaves room for the answer
    MAX_HISTORY_TURNS = 3       # previous question/answer pairs sent for follow-up context
    MAX_HISTORY_ANSWER_CHARS = 1500

    @classmethod
    def _history_messages(cls, history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Last few chat turns as role/content messages (long answers truncated to save context)."""
        msgs = []
        for m in (history or [])[-2 * cls.MAX_HISTORY_TURNS:]:
            if m.get("role") not in ("user", "assistant"):
                continue
            content = m.get("content", "")
            if m["role"] == "assistant" and len(content) > cls.MAX_HISTORY_ANSWER_CHARS:
                content = content[:cls.MAX_HISTORY_ANSWER_CHARS] + " ..."
            msgs.append({"role": m["role"], "content": content})
        # Chat templates expect the conversation to start with a user turn
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs

    @staticmethod
    def _build_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            diag_info = f" [Associated Diagram: {c['diagram_caption']}]" if c.get("has_diagram") else ""
            context_parts.append(
                f"--- [Excerpt {i} | Document: {c['document']} | Page: {c['page']}{diag_info}] ---\n{c['text']}"
            )
        context_str = "\n\n".join(context_parts)
        return (
            f"Retrieved Excerpts:\n{context_str}\n\n"
            f"Question:\n{query}\n\n"
            f"Provide a thorough, cited technical response:"
        )

    def _fit_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        token_counter=None,
        token_budget: Optional[int] = None,
        history_text: str = ""
    ) -> List[Dict[str, Any]]:
        """Drops the lowest-ranked chunks until history + prompt fit the context budget."""
        chunks = list(chunks)
        while len(chunks) > 1:
            prompt = history_text + self._build_user_prompt(query, chunks)
            too_long = len(prompt) > self.MAX_CONTEXT_CHARS or (
                token_counter is not None and token_counter(self.SYSTEM_PROMPT + prompt) > token_budget
            )
            if not too_long:
                break
            chunks.pop()
        return chunks

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", ollama_url).rstrip("/")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def get_available_ollama_models(self) -> List[str]:
        """Discovers locally installed Ollama models if Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        backend: str = "local_gguf",
        model_name_or_path: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generates grounded answer from retrieved context using genuine LLM inference.

        history: earlier chat turns ({"role", "content"}), excluding the current query.
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant documents or excerpts found in the knowledge base. Please add PDFs to `knowledge_base/` and build the index.",
                "backend": backend,
                "sources": []
            }

        history_msgs = self._history_messages(history)
        history_text = "\n".join(m["content"] for m in history_msgs)
        retrieved_chunks = self._fit_chunks(query, retrieved_chunks, history_text=history_text)

        def build_messages():
            return [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                *history_msgs,
                {"role": "user", "content": self._build_user_prompt(query, retrieved_chunks)}
            ]

        # 1. Local HuggingFace GGUF Model (Direct from models/ directory)
        if backend == "local_gguf":
            local_models = LocalGGUFEngine.get_available_models()
            if not local_models:
                return {
                    "answer": (
                        "⚠️ **No GGUF model found in `models/` directory.**\n\n"
                        "To generate answers offline, please either:\n"
                        "1. Download a model using the **'Download Model from Hugging Face'** tool in the sidebar.\n"
                        "2. Or download any `.gguf` file (e.g. from [HuggingFace](https://huggingface.co/models?search=gguf)) and drop it into `RAG_workflow/models/`."
                    ),
                    "backend": "Local GGUF (No model)",
                    "sources": retrieved_chunks
                }

            # Select model file
            if model_name_or_path:
                model_file = Path(model_name_or_path)
                if not model_file.exists():
                    model_file = MODELS_DIR / model_name_or_path
            else:
                model_file = local_models[0]

            try:
                llm = LocalGGUFEngine.get_llm(model_file, n_ctx=self.LOCAL_N_CTX)
                # Exact token check with the model's own tokenizer (64 tokens for the chat template)
                retrieved_chunks = self._fit_chunks(
                    query,
                    retrieved_chunks,
                    token_counter=lambda s: len(llm.tokenize(s.encode("utf-8"), add_bos=False)),
                    token_budget=llm.n_ctx() - self.MAX_ANSWER_TOKENS - 64 - 8 * len(history_msgs),
                    history_text=history_text
                )
                response = llm.create_chat_completion(
                    messages=build_messages(),
                    temperature=0.1,
                    max_tokens=self.MAX_ANSWER_TOKENS
                )
                raw_answer = response["choices"][0]["message"]["content"].strip()
                cleaned = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
                return {
                    "answer": cleaned,
                    "backend": f"Offline GGUF ({model_file.name})",
                    "sources": retrieved_chunks
                }
            except Exception as e:
                return {
                    "answer": f"❌ Error executing local model ({model_file.name}): {str(e)}",
                    "backend": f"Offline GGUF ({model_file.name})",
                    "sources": retrieved_chunks
                }

        # 2. Ollama local backend
        if backend == "ollama":
            model = model_name_or_path or "llama3.2"
            try:
                payload = json.dumps({
                    "model": model,
                    "messages": build_messages(),
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": self.LOCAL_N_CTX}
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.ollama_url}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=90.0) as resp:
                    res_data = json.loads(resp.read().decode())
                    raw_answer = res_data.get("message", {}).get("content", "").strip()
                    cleaned = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
                    return {
                        "answer": cleaned,
                        "backend": f"Ollama ({model})",
                        "sources": retrieved_chunks
                    }
            except Exception as e:
                return {
                    "answer": f"❌ Error querying Ollama server: {str(e)}. Make sure `ollama serve` is running.",
                    "backend": f"Ollama ({model})",
                    "sources": retrieved_chunks
                }

        # 3. Google Gemini API backend
        if backend == "gemini":
            if not self.gemini_key:
                return {
                    "answer": "⚠️ GEMINI_API_KEY is not set. Please add it to your `.env` file or enter it in the sidebar.",
                    "backend": "Google Gemini",
                    "sources": retrieved_chunks
                }
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_key)
                messages = build_messages()
                conversation = "".join(
                    f"{m['role'].title()}: {m['content']}\n\n" for m in messages[1:-1]
                )
                full_query = self.SYSTEM_PROMPT + "\n\n"
                if conversation:
                    full_query += f"Conversation so far:\n{conversation}"
                full_query += messages[-1]["content"]
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_query,
                )
                return {
                    "answer": response.text.strip(),
                    "backend": "Google Gemini",
                    "sources": retrieved_chunks
                }
            except Exception as e:
                return {
                    "answer": f"❌ Error contacting Gemini API: {str(e)}",
                    "backend": "Google Gemini",
                    "sources": retrieved_chunks
                }

        return {
            "answer": "⚠️ No valid model backend selected.",
            "backend": "None",
            "sources": retrieved_chunks
        }


# =============================================================================
# 5. STREAMLIT WEB APPLICATION
# =============================================================================

def run_streamlit_app():
    """Renders the modern Streamlit web UI."""
    import streamlit as st

    st.set_page_config(
        page_title="Document & Diagram RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Embedding model, reranker and DB client are loaded once and shared by all browser sessions
    @st.cache_resource
    def load_retriever() -> SemanticRetriever:
        return SemanticRetriever(VectorDatabase())

    retriever = load_retriever()
    vdb = retriever.vector_db
    generator = LLMGenerator()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------------------------------------------------
    # SIDEBAR: Knowledge Base & Settings
    # ---------------------------------------------------------
    with st.sidebar:
        st.title("📚 Knowledge Base")
        st.caption("Drop any PDF into `knowledge_base/` and build the index.")

        # Document Inventory
        pdf_files = list(KB_DIR.glob("*.pdf"))
        coll_count = vdb.count()

        st.metric(label="Documents in Knowledge Base", value=len(pdf_files))
        st.metric(label="Total Indexed Chunks", value=coll_count)

        if pdf_files:
            with st.expander("📄 View Files in Knowledge Base", expanded=False):
                for p in pdf_files:
                    size_kb = p.stat().st_size / 1024
                    st.write(f"- **{p.name}** ({size_kb:.1f} KB)")
        else:
            st.warning("No PDFs detected in `knowledge_base/` folder.")

        # Indexing Action Button
        if st.button("🔄 Build / Update Index", use_container_width=True, type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(ratio, text):
                progress_bar.progress(ratio)
                status_text.text(text)

            res = vdb.build_or_update(progress_callback=update_progress)
            st.success(
                f"Indexed {res['total_chunks']} chunks and {res['total_diagrams']} diagrams from "
                f"{len(res['files_processed'])} file(s); {len(res['files_skipped'])} unchanged, "
                f"{len(res['files_removed'])} removed."
            )
            time.sleep(1)
            st.rerun()

        st.divider()

        # Offline Model Management (HuggingFace GGUF models in models/)
        st.subheader("🤖 Offline LLM Models")
        local_ggufs = LocalGGUFEngine.get_available_models()

        backend_choices = []
        if local_ggufs:
            backend_choices.append("Local GGUF (from models/)")
        
        ollama_models = generator.get_available_ollama_models()
        if ollama_models:
            backend_choices.append("Local Ollama")
        if os.getenv("GEMINI_API_KEY"):
            backend_choices.append("Google Gemini")

        if not backend_choices:
            backend_choices = ["Local GGUF (from models/)"]

        selected_backend_label = st.selectbox("Active Backend", backend_choices)

        selected_model_arg = None
        if selected_backend_label == "Local GGUF (from models/)":
            backend_code = "local_gguf"
            if local_ggufs:
                model_names = [m.name for m in local_ggufs]
                selected_model_name = st.selectbox("Select Model from `models/`", model_names)
                selected_model_arg = str(MODELS_DIR / selected_model_name)
            else:
                st.warning("⚠️ No `.gguf` models found in `models/`.")
                st.info("Download a model below or drop any GGUF file into `models/`.")

            # Hugging Face Downloader
            with st.expander("⬇️ Download Model from Hugging Face", expanded=not bool(local_ggufs)):
                st.caption("Select a recommended quantized model to download into `models/`:")
                choice = st.selectbox("Recommended Models", list(POPULAR_HF_MODELS.keys()))
                model_info = POPULAR_HF_MODELS[choice]
                st.write(f"*{model_info['description']}*")

                if st.button(f"Download {model_info['filename']}", use_container_width=True):
                    with st.spinner(f"Downloading {model_info['filename']} from Hugging Face Hub (this may take several minutes)..."):
                        try:
                            saved_path = LocalGGUFEngine.download_from_hf(
                                repo_id=model_info["repo_id"],
                                filename=model_info["filename"]
                            )
                            st.success(f"Downloaded successfully: {saved_path.name}")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Download failed: {str(e)}")

        elif selected_backend_label == "Local Ollama":
            backend_code = "ollama"
            selected_model_arg = st.selectbox("Ollama Model", ollama_models)
        else:
            backend_code = "gemini"
            selected_model_arg = None

        st.divider()

        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Top Sources to Retrieve", min_value=1, max_value=10, value=4)
        use_reranker = st.checkbox("Enable FlashRank Reranker", value=HAS_FLASHRANK, disabled=not HAS_FLASHRANK)

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ---------------------------------------------------------
    # MAIN CHAT AREA
    # ---------------------------------------------------------
    st.header("📖 Multimodal Document & Diagram RAG")
    st.caption("Ask questions, explore technical concepts, and automatically retrieve referenced schematics, charts, and diagrams.")

    if not st.session_state.chat_history:
        st.info(
            "👋 **Welcome!** Ask any question about your indexed documents. "
            "Referenced diagrams, cross-sections, and schematics will be rendered directly below the generated answer."
        )

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render diagrams if present in assistant response
            if msg.get("diagrams"):
                with st.expander("🖼️ View Referenced Diagrams & Figures", expanded=True):
                    cols = st.columns(min(len(msg["diagrams"]), 3))
                    for idx, diag in enumerate(msg["diagrams"]):
                        img_path = BASE_DIR / diag["path"]
                        if img_path.exists():
                            with cols[idx % len(cols)]:
                                st.image(
                                    str(img_path),
                                    caption=f"Page {diag['page']}: {diag['caption']}",
                                    use_container_width=True
                                )

            # Render source citations
            if msg.get("sources"):
                with st.expander("📑 View Source Citations & Excerpts", expanded=False):
                    for i, s in enumerate(msg["sources"], 1):
                        score = s.get("rerank_score", s.get("similarity_score", 0.0))
                        st.markdown(f"**[{i}] {s['document']} — Page {s['page']}** *(Relevance: {score:.3f})*")
                        st.markdown(f"> {s['text']}")
                        st.divider()

    # Chat Input
    query = st.chat_input("Ask a question about your knowledge base...")
    if query:
        previous_turns = list(st.session_state.chat_history)

        # 1. Add user message to UI
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Retrieve relevant context. Short follow-ups ("what about its units?") carry
        #    little meaning alone, so they are searched together with the previous question
        retrieval_query = query
        prev_questions = [m["content"] for m in previous_turns if m["role"] == "user"]
        if prev_questions and len(query.split()) <= 8:
            retrieval_query = f"{prev_questions[-1]}\n{query}"

        with st.chat_message("assistant"):
            with st.spinner("Retrieving document excerpts & diagrams..."):
                chunks = retriever.retrieve(retrieval_query, top_k=top_k, rerank=use_reranker)

            # 3. Generate Answer
            with st.spinner("Generating grounded answer..."):
                gen_res = generator.generate(
                    query=query,
                    retrieved_chunks=chunks,
                    backend=backend_code,
                    model_name_or_path=selected_model_arg,
                    history=previous_turns
                )
                answer_text = gen_res["answer"]
                st.markdown(answer_text)

            # 4. Extract unique diagrams associated with retrieved sources
            referenced_diagrams = []
            seen_paths = set()
            for c in chunks:
                if c.get("has_diagram") and c.get("diagram_path"):
                    d_path = c["diagram_path"]
                    if d_path not in seen_paths:
                        seen_paths.add(d_path)
                        referenced_diagrams.append({
                            "path": d_path,
                            "caption": c.get("diagram_caption", "Diagram"),
                            "page": c.get("page", 1)
                        })

            # Render Diagrams
            if referenced_diagrams:
                with st.expander("🖼️ View Referenced Diagrams & Figures", expanded=True):
                    cols = st.columns(min(len(referenced_diagrams), 3))
                    for idx, diag in enumerate(referenced_diagrams):
                        img_path = BASE_DIR / diag["path"]
                        if img_path.exists():
                            with cols[idx % len(cols)]:
                                st.image(
                                    str(img_path),
                                    caption=f"Page {diag['page']}: {diag['caption']}",
                                    use_container_width=True
                                )

            # Render Sources
            if chunks:
                with st.expander("📑 View Source Citations & Excerpts", expanded=False):
                    for i, s in enumerate(chunks, 1):
                        score = s.get("rerank_score", s.get("similarity_score", 0.0))
                        st.markdown(f"**[{i}] {s['document']} — Page {s['page']}** *(Relevance: {score:.3f})*")
                        st.markdown(f"> {s['text']}")
                        st.divider()

        # Save assistant message in session history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "sources": chunks,
            "diagrams": referenced_diagrams
        })


# =============================================================================
# 6. COMMAND-LINE INTERFACE (CLI)
# =============================================================================

def run_cli():
    """Allows running ingestion and query lookups directly from terminal."""
    parser = argparse.ArgumentParser(description="Multimodal Document & Diagram RAG")
    parser.add_argument("--index", action="store_true", help="Index new/changed PDFs in knowledge_base/")
    parser.add_argument("--rebuild", action="store_true", help="Re-index all PDFs, including unchanged ones")
    parser.add_argument("--query", "-q", type=str, help="Run a query against the knowledge base")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--model-path", type=str, help="Path to local GGUF model in models/")
    parser.add_argument("--download-model", type=str, choices=list(POPULAR_HF_MODELS.keys()), help="Download recommended model from Hugging Face")
    args = parser.parse_args()

    vdb = VectorDatabase()
    retriever = SemanticRetriever(vdb)
    generator = LLMGenerator()

    if args.download_model:
        info = POPULAR_HF_MODELS[args.download_model]
        print(f"Downloading {info['filename']} from Hugging Face ({info['repo_id']})...")
        path = LocalGGUFEngine.download_from_hf(info["repo_id"], info["filename"])
        print(f"Downloaded to: {path}")
        return

    if args.index or args.rebuild:
        print("=" * 65)
        print("BUILDING / UPDATING KNOWLEDGE BASE INDEX")
        print("=" * 65)
        res = vdb.build_or_update(force=args.rebuild)
        print(f"Status: {res['status']}")
        print(f"Files processed: {len(res['files_processed'])}, "
              f"unchanged: {len(res['files_skipped'])}, removed: {len(res['files_removed'])}")
        print(f"Chunks indexed: {res.get('total_chunks', 0)}")
        print(f"Diagrams extracted: {res.get('total_diagrams', 0)}")
        print(f"Total collection count: {vdb.count()}")
        return

    if args.query:
        print("=" * 65)
        print(f"QUERY: {args.query}")
        print("=" * 65)
        chunks = retriever.retrieve(args.query, top_k=args.top_k)
        if not chunks:
            print("No matching excerpts found.")
            return

        print(f"\n[Found {len(chunks)} source excerpts]:")
        for i, c in enumerate(chunks, 1):
            diag = f" [Diagram: {c['diagram_caption']}]" if c.get("has_diagram") else ""
            print(f"  [{i}] {c['document']} (Page {c['page']}) — Score: {c['similarity_score']:.3f}{diag}")

        print("\n[Synthesizing Answer]...\n")
        # Check for local GGUF models
        local_models = LocalGGUFEngine.get_available_models()
        if args.model_path:
            chosen_model = args.model_path
            backend = "local_gguf"
        elif local_models:
            chosen_model = str(local_models[0])
            backend = "local_gguf"
        else:
            chosen_model = None
            backend = "ollama"

        res = generator.generate(args.query, chunks, backend=backend, model_name_or_path=chosen_model)
        print("-" * 65)
        print(f"BACKEND: {res['backend']}")
        print("-" * 65)
        print(res["answer"])
        print("=" * 65)
        return

    # Default action if run without arguments: launch Streamlit
    run_streamlit_app()


if __name__ == "__main__":
    if any("streamlit" in arg for arg in sys.argv) or os.getenv("STREAMLIT_RUN") == "1":
        run_streamlit_app()
    elif len(sys.argv) > 1 and (sys.argv[1].startswith("--") or sys.argv[1].startswith("-")):
        run_cli()
    else:
        run_streamlit_app()
