import gc
import os
import re
import time
import pickle
import threading
import unicodedata
import hashlib
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Optional, Any

import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http.models import Distance, PointStruct, VectorParams  # type: ignore
    QDRANT_AVAILABLE = True
except Exception:
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    PointStruct = None  # type: ignore
    VectorParams = None  # type: ignore
    QDRANT_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==============================================================================
# SECRETS MANAGER IMPORT
# ==============================================================================
try:
    from secrets_manager import get_secrets_manager
    SECRETS_MANAGER_AVAILABLE = True
except ImportError:
    SECRETS_MANAGER_AVAILABLE = False
    get_secrets_manager = None  # type: ignore

# ==============================================================================
# INTENT CLASSIFIER IMPORT
# ==============================================================================
try:
    from intent_classifier import classify_intent, Intent
    INTENT_CLASSIFIER_AVAILABLE = True
except ImportError:
    INTENT_CLASSIFIER_AVAILABLE = False
    Intent = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None

try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

def _load_env() -> None:
    load_dotenv()
    env_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.txt")
    if os.path.exists(env_txt):
        load_dotenv(env_txt, override=False)


def _configure_hf_cache() -> None:
    if os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.getenv("SENTENCE_TRANSFORMERS_HOME"):
        return
    default_cache = r"F:\huggingface_cache"
    if os.path.exists(default_cache):
        os.environ["HF_HOME"] = default_cache
        os.environ["TRANSFORMERS_CACHE"] = default_cache
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = default_cache


_load_env()
_configure_hf_cache()


@dataclass
class RAGConfig:
    """Configuration for RAG Engine."""
    # File paths
    csv_file: str = field(default_factory=lambda: os.getenv("CSV_FILE", "data/QA.csv"))
    pdf_file: str = field(default_factory=lambda: os.getenv("PDF_FILE", "data/quyche.pdf"))
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    
    # Model names
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3")
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-base")
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL") or "gemini-2.5-flash")

    # Vector DB settings
    vector_db_type: str = field(default_factory=lambda: (os.getenv("VECTOR_DB_TYPE") or "qdrant").strip().lower())
    qdrant_url: str = field(default_factory=lambda: (os.getenv("QDRANT_URL") or "http://127.0.0.1:6333").strip())
    qdrant_collection: str = field(default_factory=lambda: (os.getenv("QDRANT_COLLECTION") or "husc_rag").strip())
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or "")
    qdrant_recreate_on_build: bool = field(
        default_factory=lambda: (os.getenv("QDRANT_RECREATE_ON_BUILD") or "true").strip().lower() in ("1", "true", "yes")
    )
    
    # Retrieval settings
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K") or "5"))
    retrieve_candidates: int = field(default_factory=lambda: int(os.getenv("RETRIEVE_CANDIDATES") or "15")) # Giảm 30 -> 15 (vẫn đủ coverage, tăng tốc reranking)
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K") or "5"))
    
    # Chunking settings
    chunk_max_tokens: int = 1024 
    csv_chunk_max_tokens: int = 256 
    chunk_overlap_tokens: int = 100 
    
    # Document names (để phân biệt nguồn, giảm ảo giác trích dẫn)
    pdf_document_name: str = field(default_factory=lambda: os.getenv("PDF_DOCUMENT_NAME", "Quy chế đào tạo"))
    csv_document_name: str = field(default_factory=lambda: os.getenv("CSV_DOCUMENT_NAME", "Hỏi đáp học vụ"))
    
    # Cache settings
    answer_cache_maxsize: int = field(default_factory=lambda: int(os.getenv("ANSWER_CACHE_MAXSIZE") or "200"))
    answer_cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("ANSWER_CACHE_TTL_SECONDS") or "0"))
    answer_cache_include_history: bool = field(default_factory=lambda: os.getenv("ANSWER_CACHE_INCLUDE_HISTORY", "").lower() in ("1", "true", "yes"))
    
    # Rate limiting
    min_request_interval: float = field(default_factory=lambda: float(os.getenv("MIN_REQUEST_INTERVAL") or "2"))
    
    # Metadata boost weights
    meta_boost_article_match: float = field(default_factory=lambda: float(os.getenv("META_BOOST_ARTICLE_MATCH") or "0.45"))  # GĐ4A: Tăng 0.20 -> 0.45 (ưu tiên mạnh chunk đúng Điều)
    meta_boost_article_present: float = field(default_factory=lambda: float(os.getenv("META_BOOST_ARTICLE_PRESENT") or "0.06"))
    meta_boost_clause_match: float = field(default_factory=lambda: float(os.getenv("META_BOOST_CLAUSE_MATCH") or "0.25"))  # Tăng 0.12 -> 0.25
    meta_boost_clause_present: float = field(default_factory=lambda: float(os.getenv("META_BOOST_CLAUSE_PRESENT") or "0.05"))
    meta_boost_section_match: float = field(default_factory=lambda: float(os.getenv("META_BOOST_SECTION_MATCH") or "0.10"))
    meta_boost_section_present: float = field(default_factory=lambda: float(os.getenv("META_BOOST_SECTION_PRESENT") or "0.04"))
    meta_boost_chapter_match: float = field(default_factory=lambda: float(os.getenv("META_BOOST_CHAPTER_MATCH") or "0.10"))
    meta_boost_chapter_present: float = field(default_factory=lambda: float(os.getenv("META_BOOST_CHAPTER_PRESENT") or "0.04"))
    meta_boost_max: float = field(default_factory=lambda: float(os.getenv("META_BOOST_MAX") or "0.45"))  # Tăng 0.30 -> 0.45 (cho phép article+clause boost đủ mạnh)

    @property
    def vector_index_path(self) -> str:
        return os.path.join(self.base_dir, "vector_index.faiss")
    
    @property
    def vector_texts_path(self) -> str:
        return os.path.join(self.base_dir, "vector_texts.pkl")
    
    @property
    def bm25_index_path(self) -> str:
        return os.path.join(self.base_dir, "bm25_index.pkl")


# ==============================================================================
# TEXT PROCESSING UTILITIES
# ==============================================================================

def preprocess_text(text: str) -> str:
    """Normalize and clean text (dùng cho CSV và text thông thường)."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00ad", "")

    url_pattern = r"(https?://[^\s]+)"
    urls = re.findall(url_pattern, text)
    for i, url in enumerate(urls):
        text = text.replace(url, f"__URL_{i}__")

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text).strip()

    for i, url in enumerate(urls):
        text = text.replace(f"__URL_{i}__", url)
    return text


def preprocess_legal_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00ad", "")
    
    # Chuẩn hóa line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Nối từ bị ngắt dòng do PDF (ví dụ: "quy-\nđịnh" → "quyđịnh")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    
    # Xóa số trang rác (dòng chỉ chứa 1 số đứng riêng lẻ)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    
    # Chỉ gộp space/tab trên cùng 1 dòng, KHÔNG gộp \n
    text = re.sub(r"[ \t]+", " ", text)
    
    # Gộp 3+ dòng trống liên tiếp thành 2 dòng trống (giữ paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Thêm boundary: chèn dòng trống trước Chương / Điều / Mục
    # CHỈ khi chúng là HEADER THẬT (không đứng sau keyword tham chiếu)
    # Tham chiếu inline: "theo Điều 36", "tại Điều 5", "khoản 3 Điều 10"... → KHÔNG chèn boundary
    text = re.sub(r"(?<!\n)(Chương\s+(?:[IVXLCDM]+|\d+))", r"\n\n\1", text, flags=re.IGNORECASE)
    
    # Điều: chỉ chèn boundary nếu KHÔNG có keyword tham chiếu ngay trước
    _xref_tail_pat = re.compile(
        r'(?:theo|tại|căn\s*cứ|nêu\s*tại|quy\s*định\s*tại|thuộc|của|xem|nói\s*tại|khoản\s*\d+)\s*$',
        re.IGNORECASE,
    )
    # Dùng 2 bước: (1) tìm tất cả vị trí "Điều X" → (2) chèn \n\n cho header thật
    _dieu_positions = []  # list of (start, end) cần chèn boundary
    for m in re.finditer(r'Điều\s+\d+[\s:.]', text, flags=re.IGNORECASE):
        start = m.start()
        if start == 0 or text[start - 1] == '\n':
            continue  # đã ở đầu dòng
        prefix = text[max(0, start - 60):start]
        if _xref_tail_pat.search(prefix):
            continue  # tham chiếu chéo → bỏ qua
        _dieu_positions.append(start)
    # Chèn \n\n từ cuối lên đầu để không lệch offset
    for pos in reversed(_dieu_positions):
        text = text[:pos] + '\n\n' + text[pos:]
    
    text = re.sub(r"(?<!\n)(Mục\s+(?:[IVXLCDM]+|\d+))", r"\n\n\1", text, flags=re.IGNORECASE)
    
    # Dọn dẹp: bỏ space thừa đầu/cuối mỗi dòng
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # Gộp lại các dòng trống thừa sau khi xử lý
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# Pattern nhận diện tham chiếu chéo (dùng chung cho nhiều hàm)
_CROSSREF_KW = re.compile(
    r'(?:theo|tại|căn\s*cứ|nêu\s*tại|quy\s*định\s*tại|thuộc|của|xem|nói\s*tại)'
    r'\s*$',
    re.IGNORECASE,
)
_CLAUSE_BEFORE_ARTICLE = re.compile(
    r'khoản\s*\d+\s*$',
    re.IGNORECASE,
)


def _is_header_article(text: str, match: re.Match) -> bool:
    
    start = match.start()
    # Header thật: ở đầu text
    if start == 0:
        return True
    # Header thật: ở đầu dòng (ký tự trước là \n)
    if text[start - 1] == '\n':
        return True
    prefix_start = max(0, start - 50)
    prefix = text[prefix_start:start]
    # Tham chiếu: đứng sau keyword
    if _CROSSREF_KW.search(prefix):
        return False
    # Tham chiếu: đứng sau "khoản N" (pattern: khoản 5 Điều 36)
    if _CLAUSE_BEFORE_ARTICLE.search(prefix):
        return False
    return True


def extract_metadata(text: str) -> dict:
    lower = (text or "").lower()
    original_text = text or ""

    article = None
    clause = None
    section = None
    chapter = None

    all_article_matches = list(re.finditer(r'\bđiều\s*(\d+)\b', lower))
    header_article_match = None
    for m in all_article_matches:
        if _is_header_article(original_text, m):
            header_article_match = m
            break  
    
    if header_article_match:
        try:
            article = int(header_article_match.group(1))
        except Exception:
            pass
    elif all_article_matches:
        try:
            article = int(all_article_matches[0].group(1))
        except Exception:
            pass

    inline_match = re.search(
        r'\bđiều\s*(\d+)(.{0,200}?)khoản\s*(\d+)\b', lower
    )
    if inline_match:
        middle_text = inline_match.group(2)
        # Nếu giữa "Điều X" và "khoản Y" có 1 "Điều" khác → hủy match
        if not re.search(r'\bđiều\s*\d+', middle_text):
            try:
                inline_article = int(inline_match.group(1))
                # Chỉ cập nhật clause, giữ article từ header nếu có
                clause = int(inline_match.group(3))
                if article is None:
                    article = inline_article
            except Exception:
                pass

    if clause is None and article is not None:
        clause_matches = re.findall(r"^(\d+)\.\s+", original_text, re.MULTILINE)
        if clause_matches:
            try:
                clause = int(clause_matches[-1])
            except Exception:
                pass

    if clause is None:
        m = re.search(r"(?:^|\b)khoản\s*(\d+)\b", lower)
        if m:
            try:
                clause = int(m.group(1))
            except Exception:
                pass
    m = re.search(r"(?:^|\b)mục\s*([ivxlcdm]+|\d+)\b", lower)
    if m:
        section = m.group(1).upper()

    m = re.search(r"(?:^|\b)chương\s*([ivxlcdm]+|\d+)\b", lower)
    if m:
        chapter = m.group(1).upper()

    return {"chapter": chapter, "section": section, "article": article, "clause": clause}


def chunk_text(text: str, max_tokens: int = 256, overlap_tokens: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            decoded = enc.decode(chunk_tokens).strip()
            if decoded:
                chunks.append(decoded)
            start = end - overlap_tokens if end < len(tokens) else end
            if start >= len(tokens) - overlap_tokens and end >= len(tokens):
                break
        return chunks
    except Exception:
        sentences = re.split(r"(?<=[.!?]) +", text)
        chunks = []
        buffer = ""
        max_len = 350
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(buffer) + len(sent) < max_len:
                buffer += " " + sent
            else:
                chunks.append(buffer.strip())
                buffer = sent
        if buffer:
            chunks.append(buffer.strip())
        return chunks


def chunk_text_hierarchical(text: str, max_tokens: int = 512) -> list[str]:

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    def _count_tokens(s: str) -> int:
        if enc:
            return len(enc.encode(s))
        return len(s) // 4

    # Split theo Điều trước
    parts = re.split(r'(Điều\s+\d+[:\.])', text, flags=re.IGNORECASE)

    # Nếu không có cấu trúc Điều → fallback
    if len(parts) <= 1:
        return chunk_text(text, max_tokens, max(max_tokens // 5, 20))

    chunks: list[str] = []

    # parts[0] là phần trước Điều đầu tiên (nếu có)
    preamble = parts[0].strip()
    if preamble and _count_tokens(preamble) > 10:
        chunks.append(preamble)

    # Iterate qua các cặp (header, content)
    i = 1
    while i < len(parts):
        header = parts[i] if i < len(parts) else ""
        content = parts[i + 1] if i + 1 < len(parts) else ""
        article_text = (header + content).strip()
        i += 2

        if not article_text:
            continue

        tokens_count = _count_tokens(article_text)

        # Nếu một Điều vừa đủ → giữ nguyên
        if tokens_count <= max_tokens:
            chunks.append(article_text)
        else:
            # Nếu quá dài → split theo Khoản (pattern: "1. ", "2. ", etc.)
            clause_parts = re.split(r'(\d+\.\s)', article_text)
            temp_chunk = header.strip()

            j = 1
            while j < len(clause_parts):
                clause_num = clause_parts[j] if j < len(clause_parts) else ""
                clause_content = clause_parts[j + 1] if j + 1 < len(clause_parts) else ""
                j += 2

                test_chunk = temp_chunk + " " + clause_num + clause_content
                if _count_tokens(test_chunk) <= max_tokens:
                    temp_chunk = test_chunk
                else:
                    if temp_chunk.strip():
                        chunks.append(temp_chunk.strip())
                    temp_chunk = header.strip() + " " + clause_num + clause_content

            if temp_chunk.strip():
                chunks.append(temp_chunk.strip())

    return chunks if chunks else chunk_text(text, max_tokens, max(max_tokens // 5, 20))


def chunk_pdf_by_structure(
    pdf_text: str,
    max_tokens: int = 256,
    document_name: str = "Quy chế đào tạo",
    doc_ranges: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Structure-aware chunking for legal/student-handbook PDFs.

    Core rule:
    - Split by real headings in source text (Chương / Mục / Điều / heading La Mã / tiêu đề),
      not only by Điều, to avoid attaching unrelated sections to the nearest Điều.
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    def count_tokens(s: str) -> int:
        if enc:
            return len(enc.encode(s))
        return len(s) // 4

    def split_by_sentences(text: str, max_tok: int) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: list[str] = []
        buffer = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if count_tokens((buffer + " " + sent).strip()) <= max_tok:
                buffer = (buffer + " " + sent).strip()
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sent
        if buffer:
            chunks.append(buffer)
        return chunks if chunks else [text]

    def _looks_like_title_line(line: str) -> bool:
        line = (line or "").strip()
        if re.match(r'^\[\[PAGE:\d+\]\]$', line):
            return False
        if len(line) < 6 or len(line) > 160:
            return False
        if re.match(r'^(Chương|Mục|Điều)\b', line, flags=re.IGNORECASE):
            return False
        if re.match(r'^\d+\.\s', line):
            return False
        letters = [ch for ch in line if ch.isalpha()]
        if len(letters) < 4:
            return False
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        return upper_ratio >= 0.82

    def _parse_chapter_value(chapter_ctx: str) -> str | None:
        if not chapter_ctx:
            return None
        m = re.search(r'Chương\s+([IVXLCDM]+|\d+)', chapter_ctx, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    def _parse_section_value(section_ctx: str) -> str | None:
        if not section_ctx:
            return None
        m = re.search(r'Mục\s+([IVXLCDM]+|\d+)', section_ctx, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.match(r'\s*([IVXLCDM]+)\.\s+', section_ctx, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    records: list[dict] = []
    doc_ranges = doc_ranges or []

    page_markers = [(m.start(), int(m.group(1))) for m in re.finditer(r'\[\[PAGE:(\d+)\]\]', pdf_text)]

    def _page_at(pos: int) -> int | None:
        page_val = None
        for marker_pos, marker_page in page_markers:
            if marker_pos <= pos:
                page_val = marker_page
            else:
                break
        return page_val

    def _resolve_doc_range(page_no: int | None) -> tuple[str, str]:
        if page_no is None:
            return document_name, document_name
        for dr in doc_ranges:
            s = dr.get("start_page")
            e = dr.get("end_page")
            if isinstance(s, int) and isinstance(e, int) and s <= page_no <= e:
                fam = str(dr.get("doc_family") or document_name)
                title = str(dr.get("source_title") or fam)
                return fam, title
        return document_name, document_name

    # Recover heading boundaries if line breaks were previously flattened.
    pdf_text = re.sub(r'(?<=[.?!])\s+(Điều\s+\d+[\s:.])', r'\n\n\1', pdf_text, flags=re.IGNORECASE)
    pdf_text = re.sub(r'(?<=[.?!])\s+(Chương\s+(?:[IVXLCDM]+|\d+))', r'\n\n\1', pdf_text, flags=re.IGNORECASE)
    pdf_text = re.sub(r'(?<=[.?!])\s+(Mục\s+(?:[IVXLCDM]+|\d+))', r'\n\n\1', pdf_text, flags=re.IGNORECASE)
    pdf_text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', pdf_text)

    chapter_pattern = re.compile(r'^\s*(Chương\s+(?:[IVXLCDM]+|\d+)[^\n]*)', re.MULTILINE | re.IGNORECASE)
    section_pattern = re.compile(r'^\s*(Mục\s+(?:[IVXLCDM]+|\d+)[^\n]*)', re.MULTILINE | re.IGNORECASE)
    article_pattern = re.compile(r'^\s*(Điều\s+(\d+)[\s:\.][^\n]*)', re.MULTILINE | re.IGNORECASE)
    roman_heading_pattern = re.compile(r'^\s*([IVXLCDM]+)\.\s+([^\n]{3,160})', re.MULTILINE | re.IGNORECASE)

    heading_candidates: list[dict] = []

    for m in chapter_pattern.finditer(pdf_text):
        heading_candidates.append({"pos": m.start(), "header": m.group(1).strip(), "type": "chapter", "priority": 0})
    for m in section_pattern.finditer(pdf_text):
        heading_candidates.append({"pos": m.start(), "header": m.group(1).strip(), "type": "section", "priority": 1})
    for m in article_pattern.finditer(pdf_text):
        heading_candidates.append(
            {
                "pos": m.start(),
                "header": m.group(1).strip(),
                "type": "article",
                "article": int(m.group(2)),
                "priority": 2,
            }
        )
    for m in roman_heading_pattern.finditer(pdf_text):
        # Keep as section-like heading only when it is not a clause line.
        if re.match(r'^\s*[IVXLCDM]+\.\s+\d+', m.group(0).strip(), flags=re.IGNORECASE):
            continue
        heading_candidates.append({"pos": m.start(), "header": m.group(0).strip(), "type": "roman_section", "priority": 3})

    for m in re.finditer(r'(?m)^([^\n]+)$', pdf_text):
        line = m.group(1).strip()
        if not line:
            continue
        if chapter_pattern.match(line) or section_pattern.match(line) or article_pattern.match(line) or roman_heading_pattern.match(line):
            continue
        if _looks_like_title_line(line):
            heading_candidates.append({"pos": m.start(1), "header": line, "type": "title", "priority": 4})

    heading_candidates.sort(key=lambda x: (x["pos"], x["priority"]))

    headings: list[dict] = []
    for cand in heading_candidates:
        if not headings:
            headings.append(cand)
            continue
        prev = headings[-1]
        # Multiple regexes can match the same line. Keep the highest-priority heading.
        if abs(int(cand["pos"]) - int(prev["pos"])) <= 1:
            if int(cand["priority"]) < int(prev["priority"]):
                headings[-1] = cand
            continue
        headings.append(cand)

    if not headings:
        chunks = chunk_text(preprocess_text(pdf_text), max_tokens, 50)
        for chunk in chunks:
            meta = extract_metadata(chunk)
            fam, title = _resolve_doc_range(_page_at(0))
            records.append(
                {
                    "question": "",
                    "text": chunk,
                    "source": "quyche.pdf",
                    "document_name": document_name,
                    "header_span": "",
                    "structure_type": "fallback",
                    "section_title": "",
                    "doc_family": fam,
                    "source_title": title,
                    "page_start": None,
                    "page_end": None,
                    **meta,
                }
            )
        return records

    chapter_positions = []
    for h in headings:
        if h["type"] != "chapter":
            continue
        h_pos = int(h["pos"])
        h_page = _page_at(h_pos)
        h_family, _ = _resolve_doc_range(h_page)
        chapter_positions.append((h_pos, str(h["header"]), h_family))
    section_positions = [
        (
            int(h["pos"]),
            str(h["header"]),
            _resolve_doc_range(_page_at(int(h["pos"])))[0],
        )
        for h in headings
        if h["type"] in {"section", "roman_section"}
    ]

    def _get_latest_at(pos: int, arr: list[tuple[int, str, str]], family: str) -> str:
        for i in range(len(arr) - 1, -1, -1):
            if arr[i][0] <= pos and arr[i][2] == family:
                return arr[i][1]
        return ""

    def _append_record(
        content: str,
        chapter_ctx: str,
        section_ctx: str,
        structure_type: str,
        header_span: str,
        article_num: int | None,
        page_start: int | None,
        page_end: int | None,
        doc_family: str,
        source_title: str,
        clause_num: int | None = None,
    ) -> None:
        chapter_val = _parse_chapter_value(chapter_ctx)
        section_val = _parse_section_value(section_ctx)
        section_title = section_ctx or (header_span if structure_type in {"title", "chapter"} else "")
        cleaned_content = re.sub(r'\[\[PAGE:\d+\]\]', ' ', content)
        records.append(
            {
                "question": "",
                "text": preprocess_text(cleaned_content),
                "source": "quyche.pdf",
                "document_name": document_name,
                "chapter": chapter_val,
                "section": section_val,
                "article": article_num,
                "clause": clause_num,
                "header_span": header_span,
                "structure_type": structure_type,
                "section_title": section_title,
                "doc_family": doc_family,
                "source_title": source_title,
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    for i, heading in enumerate(headings):
        pos = int(heading["pos"])
        next_pos = int(headings[i + 1]["pos"]) if i + 1 < len(headings) else len(pdf_text)
        segment_text = pdf_text[pos:next_pos].strip()
        if not segment_text:
            continue

        page_start = _page_at(pos)
        page_end = _page_at(max(pos, next_pos - 1))
        doc_family, source_title = _resolve_doc_range(page_start)

        chapter_ctx = _get_latest_at(pos, chapter_positions, doc_family)
        section_ctx = _get_latest_at(pos, section_positions, doc_family)

        heading_type = str(heading["type"])
        header_span = str(heading["header"]).strip()
        article_num = int(heading["article"]) if heading_type == "article" and heading.get("article") is not None else None

        if heading_type == "chapter":
            chapter_ctx = header_span
            section_ctx = ""
        elif heading_type in {"section", "roman_section"}:
            section_ctx = header_span

        structure_type_map = {
            "article": "article",
            "chapter": "chapter",
            "section": "section",
            "roman_section": "section",
            "title": "title",
        }
        structure_type = structure_type_map.get(heading_type, "unknown")

        prefix_parts: list[str] = []
        lower_segment = segment_text.lower()
        if chapter_ctx and chapter_ctx.lower() not in lower_segment[:200]:
            prefix_parts.append(chapter_ctx)
        if section_ctx and section_ctx.lower() not in lower_segment[:200] and section_ctx != chapter_ctx:
            prefix_parts.append(section_ctx)
        prefix = "\n".join(prefix_parts)

        full_text = f"{prefix}\n{segment_text}".strip() if prefix else segment_text
        token_count = count_tokens(full_text)

        if structure_type == "article":
            if token_count <= max_tokens:
                _append_record(
                    full_text, chapter_ctx, section_ctx, structure_type, header_span, article_num,
                    page_start, page_end, doc_family, source_title, None
                )
                continue

            reserved_tokens = count_tokens(prefix) + count_tokens(header_span) + 20
            effective_max = max(120, max_tokens - reserved_tokens)
            clause_splits = re.split(r'(?=(?:^|\n)\s*\d+\.\s)', segment_text)
            chunk_units: list[tuple[str, int | None]] = []

            if len(clause_splits) > 1:
                for part in clause_splits:
                    part = part.strip()
                    if not part:
                        continue
                    m_clause = re.match(r'^(\d+)\.\s', part)
                    clause_num = int(m_clause.group(1)) if m_clause else None
                    if count_tokens(part) <= effective_max:
                        chunk_units.append((part, clause_num))
                    else:
                        for ss in split_by_sentences(part, effective_max):
                            chunk_units.append((ss, clause_num))
            else:
                for ss in split_by_sentences(segment_text, effective_max):
                    m_clause = re.search(r'(?:^|\n)\s*(\d+)\.\s', ss)
                    clause_num = int(m_clause.group(1)) if m_clause else None
                    chunk_units.append((ss, clause_num))

            for sub_chunk, clause_num in chunk_units:
                if header_span and header_span not in sub_chunk:
                    chunk_content = f"{header_span}... {sub_chunk}"
                else:
                    chunk_content = sub_chunk
                chunk_text_final = f"{prefix}\n{chunk_content}".strip() if prefix else chunk_content
                _append_record(
                    chunk_text_final, chapter_ctx, section_ctx, structure_type, header_span, article_num,
                    page_start, page_end, doc_family, source_title, clause_num
                )
            continue

        if token_count <= max_tokens:
            _append_record(
                full_text, chapter_ctx, section_ctx, structure_type, header_span, None,
                page_start, page_end, doc_family, source_title, None
            )
            continue

        reserved_tokens = count_tokens(prefix) + 20
        effective_max = max(120, max_tokens - reserved_tokens)
        for sub_chunk in split_by_sentences(segment_text, effective_max):
            chunk_text_final = f"{prefix}\n{sub_chunk}".strip() if prefix else sub_chunk
            clause_match = re.search(r'(?:^|\n)\s*(\d+)\.\s', sub_chunk)
            clause_num = int(clause_match.group(1)) if clause_match else None
            _append_record(
                chunk_text_final, chapter_ctx, section_ctx, structure_type, header_span, None,
                page_start, page_end, doc_family, source_title, clause_num
            )

    return records

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using available backend."""
    backend = (os.getenv("PDF_BACKEND") or "auto").strip().lower()
    extract_tables = (os.getenv("PDF_EXTRACT_TABLES") or "").strip().lower() in ("1", "true", "yes")

    def _postprocess(raw: str) -> str:
        raw = raw or ""
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        raw = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", raw)
        # Dùng preprocess_legal_text thay vì preprocess_text
        # để GIỮ NGUYÊN \n cho structure-aware chunking
        return preprocess_legal_text(raw)

    last_error: Exception | None = None

    if backend in ("auto", "pymupdf", "fitz"):
        if fitz is not None:
            try:
                parts: list[str] = []
                doc = fitz.open(pdf_path)
                try:
                    for i, page in enumerate(doc):
                        parts.append(f"[[PAGE:{i+1}]]\n{page.get_text('text') or ''}")
                finally:
                    doc.close()
                return _postprocess("\n".join(parts))
            except Exception as e:
                last_error = e
        elif backend != "auto":
            raise RuntimeError("PDF_BACKEND='pymupdf' nhưng chưa cài PyMuPDF (import fitz thất bại).")

    if backend in ("auto", "pdfplumber"):
        if pdfplumber is not None:
            try:
                parts = []
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        parts.append(f"[[PAGE:{i+1}]]\n{page.extract_text() or ''}")
                        if extract_tables:
                            tables = page.extract_tables() or []
                            for table in tables:
                                rows = []
                                for row in table or []:
                                    rows.append("\t".join("" if c is None else str(c) for c in row))
                                if rows:
                                    parts.append("\n".join(rows))
                return _postprocess("\n".join(parts))
            except Exception as e:
                last_error = e
        elif backend != "auto":
            raise RuntimeError("PDF_BACKEND='pdfplumber' nhưng chưa cài pdfplumber.")

    if backend in ("auto", "pypdf2"):
        if PdfReader is not None:
            try:
                text = ""
                with open(pdf_path, "rb") as f:
                    reader = PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        content = page.extract_text()
                        if content:
                            text += f"[[PAGE:{i+1}]]\n{content}\n"
                return _postprocess(text)
            except Exception as e:
                last_error = e
        else:
            raise RuntimeError("PDF_BACKEND='pypdf2' nhưng chưa cài PyPDF2.")

    if backend not in ("auto", "pymupdf", "fitz", "pdfplumber", "pypdf2"):
        raise ValueError("PDF_BACKEND không hợp lệ. Dùng: auto | pymupdf | pdfplumber | pypdf2.")

    if last_error is not None:
        raise last_error
    raise RuntimeError("Không có backend đọc PDF khả dụng (cài PyMuPDF hoặc pdfplumber, hoặc PyPDF2).")


def _guess_doc_family_from_title(title: str) -> str:
    t = (title or "").strip().lower()
    if "công tác sinh viên" in t:
        return "Quy chế Công tác sinh viên"
    if "học vụ" in t:
        return "Quy chế Học vụ"
    if "học bổng" in t:
        return "Quy định học bổng"
    if "chính sách" in t or "miễn, giảm học phí" in t or "miễn giảm học phí" in t:
        return "Chế độ, chính sách sinh viên"
    return title.strip() or "Quy chế đào tạo"


def infer_pdf_doc_ranges(pdf_path: str) -> list[dict]:
   
    entries: list[tuple[int, str]] = []
    page_count = 0

    if PdfReader is not None:
        try:
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            scan_from = max(0, page_count - 6)
            for p in range(scan_from, page_count):
                text = reader.pages[p].extract_text() or ""
                for raw_line in text.splitlines():
                    line = re.sub(r'[\s\-–—_\.]{2,}', ' ', (raw_line or "")).strip()
                    m = re.match(r'^(\d+)\.\s+(.+?)\s+(\d{1,3})$', line)
                    if not m:
                        continue
                    idx = int(m.group(1))
                    title = m.group(2).strip()
                    start_page = int(m.group(3))
                    if idx >= 1 and start_page >= 1:
                        entries.append((start_page, title))
        except Exception:
            entries = []

    # Keep unique starts
    dedup: dict[int, str] = {}
    for start_page, title in entries:
        if start_page not in dedup:
            dedup[start_page] = title
    sorted_entries = sorted(dedup.items(), key=lambda x: x[0])

    ranges: list[dict] = []
    if sorted_entries:
        for i, (start_page, title) in enumerate(sorted_entries):
            end_page = (sorted_entries[i + 1][0] - 1) if i + 1 < len(sorted_entries) else page_count
            if end_page < start_page:
                end_page = start_page
            ranges.append(
                {
                    "start_page": start_page,
                    "end_page": end_page,
                    "source_title": title,
                    "doc_family": _guess_doc_family_from_title(title),
                }
            )
        # Keep only relevant handbook sections with legal/academic content.
        filtered = []
        for r in ranges:
            fam = str(r.get("doc_family", "")).lower()
            if any(k in fam for k in ["công tác sinh viên", "học vụ", "học bổng", "chính sách"]):
                filtered.append(r)
        if filtered:
            return filtered

    # Fallback for the known handbook layout.
    base = os.path.basename(pdf_path).lower()
    if "quyche" in base and page_count >= 99:
        return [
            {
                "start_page": 6,
                "end_page": 41,
                "source_title": "Các quy định liên quan đến Công tác sinh viên",
                "doc_family": "Quy chế Công tác sinh viên",
            },
            {
                "start_page": 42,
                "end_page": 85,
                "source_title": "Quy chế học vụ",
                "doc_family": "Quy chế Học vụ",
            },
            {
                "start_page": 86,
                "end_page": 98,
                "source_title": "Chế độ chính sách đối với sinh viên",
                "doc_family": "Chế độ, chính sách sinh viên",
            },
            {
                "start_page": 99,
                "end_page": page_count,
                "source_title": "Học bổng sinh viên",
                "doc_family": "Quy định học bổng",
            },
        ]

    return []


# ==============================================================================
# CONTEXT VALIDATION & ANTI-HALLUCINATION
# ==============================================================================

class ContextValidator:
    """Validate context relevance and detect potential hallucination risks."""
    
    # Keywords indicating the query needs specific information
    SPECIFIC_INFO_KEYWORDS = [
        "bao nhiêu", "mấy", "số lượng", "thời gian", "ngày", "tháng", "năm",
        "điều kiện", "yêu cầu", "quy định", "thủ tục", "hồ sơ", "giấy tờ",
        "điểm", "gpa", "tín chỉ", "học phí", "mức", "tỷ lệ", "phần trăm"
    ]
    
    
    @staticmethod
    def calculate_relevance_score(query: str, context_text: str) -> float:
        """Calculate how relevant the context is to the query."""
        query_lower = query.lower()
        context_lower = context_text.lower()
        
        # Extract key terms from query
        query_tokens = set(re.findall(r'\b\w{2,}\b', query_lower))
        context_tokens = set(re.findall(r'\b\w{2,}\b', context_lower))
        
        if not query_tokens:
            return 0.0
        
        # Calculate overlap
        overlap = query_tokens & context_tokens
        overlap_ratio = len(overlap) / len(query_tokens)
        
        # Check for specific entity matches (điều, khoản, etc.)
        entity_bonus = 0.0
        
        # Match "điều X"
        article_match = re.search(r'điều\s*(\d+)', query_lower)
        if article_match:
            article_num = article_match.group(1)
            if re.search(rf'điều\s*{article_num}\b', context_lower):
                entity_bonus += 0.3
        
        # Match "khoản X"
        clause_match = re.search(r'khoản\s*(\d+)', query_lower)
        if clause_match:
            clause_num = clause_match.group(1)
            if re.search(rf'khoản\s*{clause_num}\b', context_lower):
                entity_bonus += 0.2
        
        # Match chapter/section
        chapter_match = re.search(r'chương\s*([ivxlcdm]+)', query_lower)
        if chapter_match:
            chapter_id = chapter_match.group(1)
            if re.search(rf'chương\s*{chapter_id}\b', context_lower, re.IGNORECASE):
                entity_bonus += 0.2
        
        return min(1.0, overlap_ratio * 0.7 + entity_bonus)
    
    @staticmethod
    def needs_specific_data(query: str) -> bool:
        """Check if query requires specific numerical/factual data."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in ContextValidator.SPECIFIC_INFO_KEYWORDS)
    
    @staticmethod
    def check_context_sufficiency(query: str, contexts: list[dict], threshold: float = 0.3) -> dict:
        """
        Check if retrieved contexts are sufficient to answer the query.
        
        Returns:
            dict with keys:
                - is_sufficient: bool
                - confidence: float
                - missing_info: list of what might be missing
                - recommendation: str
        """
        if not contexts:
            return {
                "is_sufficient": False,
                "confidence": 0.0,
                "missing_info": ["Không tìm thấy ngữ cảnh liên quan"],
                "recommendation": "ask_clarification"
            }
        
        # Combine all context texts
        combined_context = "\n".join([
            f"{c.get('question', '')} {c.get('text', '')}" 
            for c in contexts
        ])
        
        # Calculate overall relevance
        relevance = ContextValidator.calculate_relevance_score(query, combined_context)
        
        # Get top scores
        top_scores = [c.get('final_score', c.get('rerank_score', 0)) for c in contexts[:3]]
        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0
        
        missing_info = []
        
        # Check for specific requirements
        if ContextValidator.needs_specific_data(query):
            # Look for numbers in context
            has_numbers = bool(re.search(r'\d+', combined_context))
            if not has_numbers:
                missing_info.append("Thiếu dữ liệu số cụ thể")
        
        # Check article/clause presence
        article_match = re.search(r'điều\s*(\d+)', query.lower())
        if article_match:
            article_num = article_match.group(1)
            if not re.search(rf'điều\s*{article_num}\b', combined_context.lower()):
                missing_info.append(f"Không tìm thấy Điều {article_num}")
        
        # Determine sufficiency
        is_sufficient = relevance >= threshold and avg_top_score >= 0.4 and len(missing_info) == 0
        
        # Determine confidence level
        if relevance >= 0.6 and avg_top_score >= 0.6:
            confidence = 0.9
        elif relevance >= 0.4 and avg_top_score >= 0.4:
            confidence = 0.7
        elif relevance >= 0.2:
            confidence = 0.5
        else:
            confidence = 0.3
        
        # Recommendation
        if is_sufficient:
            recommendation = "answer_normally"
        elif confidence >= 0.5:
            recommendation = "answer_with_disclaimer"
        else:
            recommendation = "ask_clarification"
        
        return {
            "is_sufficient": is_sufficient,
            "confidence": confidence,
            "relevance_score": relevance,
            "avg_retrieval_score": avg_top_score,
            "missing_info": missing_info,
            "recommendation": recommendation
        }


# ==============================================================================
# RAG ENGINE CLASS
# ==============================================================================

class RAGEngine:
    """
    Main RAG Engine class encapsulating all retrieval and generation logic.
    
    Usage:
        engine = RAGEngine()
        engine.initialize()
        answer = engine.generate_answer("Điều 5 nói gì?")
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG Engine with configuration."""
        self.config = config or RAGConfig()
        
        # Models
        self._embedder: Optional[SentenceTransformer] = None
        self._reranker: Optional[CrossEncoder] = None
        self._gemini_model: Optional[Any] = None
        
        # Database
        self._index: Optional[Any] = None
        self._qdrant_client: Optional[Any] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []
        self._records: list[dict] = []
        # Cross-document mapping indexes
        self._record_key_to_idx: dict[str, int] = {}
        self._article_to_indices: dict[str, list[int]] = {}
        self._family_article_to_indices: dict[tuple[str, str], list[int]] = {}
        self._article_families: dict[str, set[str]] = {}
        self._family_display_names: dict[str, str] = {}
        self._xref_links: dict[int, list[int]] = {}
        self._graduation_scale_indices: list[int] = []
        
        # State
        self._gemini_ready: bool = False
        self._initialized: bool = False
        self._last_api_call: float = 0.0
        
        # Answer cache
        self._answer_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._answer_cache_hits: int = 0
        self._answer_cache_misses: int = 0
        self._cache_lock = threading.Lock()
        # Bump version khi thay đổi prompt/logic để tránh dùng lại câu trả lời cache cũ.
        self._answer_policy_version: str = "2026-02-21-rag-format-polish-v1"
        
        # Query embedding cache (LRU, tránh encode lại query giống nhau)
        self._query_embedding_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._query_cache_maxsize: int = 100
        
        # Reranker auto-unload timer
        self._reranker_last_used: float = 0.0
        self._reranker_unload_timer: Optional[threading.Timer] = None
        self._reranker_unload_delay: int = 300  # 5 phút không dùng → unload
        
        # Init lock
        self._init_lock = threading.Lock()

    # ==========================================================================
    # Properties
    # ==========================================================================
    
    @property
    def gemini_ready(self) -> bool:
        return self._gemini_ready
    
    @property
    def embedder(self) -> Optional[SentenceTransformer]:
        return self._embedder
    
    @property
    def index(self) -> Optional[Any]:
        return self._index
    
    @property
    def records(self) -> list[dict]:
        return self._records
    
    @property
    def bm25_index(self) -> Optional[BM25Okapi]:
        return self._bm25_index

    @property
    def vector_db_type(self) -> str:
        db_type = (self.config.vector_db_type or "faiss").strip().lower()
        return db_type if db_type in ("faiss", "qdrant") else "faiss"

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def initialize(self, load_db: bool = True) -> None:
        """Initialize the RAG engine (load models and database)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                if self._embedder is None:
                    print(f"📥 Loading embedding model: {self.config.embedding_model}")
                    self._embedder = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                print(f"❌ Lỗi khởi tạo embedding model: {e}")
                self._embedder = None
            try:
                self._load_reranker()
            except Exception:
                pass
            if load_db:
                self.load_database()
            self._initialized = True

    def _load_reranker(self) -> Optional[CrossEncoder]:
        """Load reranker on-demand với auto-unload sau 5 phút idle."""
        try:
            if self._reranker is None:
                print("🔄 Loading reranker model...")
                self._reranker = CrossEncoder(self.config.reranker_model, max_length=512)
            
            # Cập nhật thời gian sử dụng & đặt timer unload
            self._reranker_last_used = time.time()
            if self._reranker_unload_timer is not None:
                self._reranker_unload_timer.cancel()
            self._reranker_unload_timer = threading.Timer(
                self._reranker_unload_delay, self._auto_unload_reranker
            )
            self._reranker_unload_timer.daemon = True
            self._reranker_unload_timer.start()
            
            return self._reranker
        except Exception as e:
            print(f"⚠️ Không thể load reranker: {e}")
            return None

    def _auto_unload_reranker(self) -> None:
        """Auto-unload reranker nếu idle quá lâu (~500MB RAM)."""
        if self._reranker is not None:
            idle_time = time.time() - self._reranker_last_used
            if idle_time >= self._reranker_unload_delay:
                print(f"🧹 Auto-unloading reranker (idle {idle_time:.0f}s). Tiết kiệm ~500MB RAM.")
                del self._reranker
                self._reranker = None
                gc.collect()

    def _load_embedder(self) -> SentenceTransformer:
        """Load embedder on-demand (lazy loading sau khi bị unload)."""
        if self._embedder is None:
            print(f"📥 Loading embedding model: {self.config.embedding_model}")
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    # ==========================================================================
    # Database Operations
    # ==========================================================================
    
    def _record_index_text(self, record: dict) -> str:
        """Get searchable text from a record."""
        f = preprocess_text(str(record.get("doc_family", "") or "")).strip()
        st = preprocess_text(str(record.get("source_title", "") or "")).strip()
        d = preprocess_text(str(record.get("document_name", "") or "")).strip()
        h = preprocess_text(str(record.get("header_span", "") or "")).strip()
        s = preprocess_text(str(record.get("section_title", "") or "")).strip()
        q = preprocess_text(str(record.get("question", "") or "")).strip()
        t = preprocess_text(str(record.get("text", "") or "")).strip()
        parts = [p for p in [f, st, d, h, s, q, t] if p]
        return "\n".join(parts)

    def _is_qdrant_backend(self) -> bool:
        return self.vector_db_type == "qdrant"

    def _init_qdrant_client(self) -> Optional[Any]:
        if not self._is_qdrant_backend():
            return None
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client chưa được cài đặt. Hãy cài: pip install qdrant-client")
        if self._qdrant_client is not None:
            return self._qdrant_client
        kwargs: dict[str, Any] = {"url": self.config.qdrant_url}
        if self.config.qdrant_api_key:
            kwargs["api_key"] = self.config.qdrant_api_key
        self._qdrant_client = QdrantClient(**kwargs)
        return self._qdrant_client

    def _ensure_qdrant_collection(self, vector_size: int, recreate: bool = False) -> None:
        client = self._init_qdrant_client()
        if client is None:
            raise RuntimeError("Qdrant client chưa sẵn sàng")
        name = self.config.qdrant_collection
        if recreate:
            client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def _make_qdrant_payload(self, record: dict, idx: int) -> dict:
        payload: dict[str, Any] = {"record_idx": int(idx)}
        for key in (
            "source",
            "document_name",
            "source_title",
            "doc_family",
            "chapter",
            "section",
            "article",
            "clause",
            "header_span",
            "structure_type",
            "page_start",
            "page_end",
        ):
            value = record.get(key)
            if value is not None and value != "":
                payload[key] = value
        return payload

    def _normalize_family_name(self, family: str) -> str:
        return re.sub(r"\s+", " ", str(family or "").strip().lower())

    def _make_record_key(self, record: dict) -> str:
        key_payload = f"{record.get('source','')}|{record.get('question','')}|{record.get('text','')}"
        return hashlib.sha1(key_payload.encode("utf-8", errors="ignore")).hexdigest()

    def _extract_inline_references(self, text: str) -> list[dict]:
        refs: list[dict] = []
        lower = (text or "").lower()
        for m in re.finditer(r'điều\s+(\d+)', lower):
            # Chỉ coi là tham chiếu chéo khi có tín hiệu ngôn ngữ trước đó.
            prefix = lower[max(0, m.start() - 35):m.start()]
            if not re.search(r'(theo|tại|khoản|điểm|căn\s*cứ|quy\s*định|nêu\s*tại)', prefix):
                continue
            article = m.group(1)
            clause = None
            m_clause = re.search(r'khoản\s+(\d+)\s*$', prefix)
            if m_clause:
                clause = m_clause.group(1)
            refs.append({"article": article, "clause": clause})
        return refs

    def _build_cross_document_mapping(self) -> None:
        """Build indices that connect clauses/articles across handbook sections."""
        self._record_key_to_idx = {}
        self._article_to_indices = {}
        self._family_article_to_indices = {}
        self._article_families = {}
        self._family_display_names = {}
        self._xref_links = {}
        self._graduation_scale_indices = []

        for idx, rec in enumerate(self._records):
            if not isinstance(rec, dict):
                continue
            key = self._make_record_key(rec)
            self._record_key_to_idx[key] = idx

            family_raw = str(
                rec.get("doc_family")
                or rec.get("source_title")
                or rec.get("document_name")
                or "văn bản liên quan"
            ).strip()
            family_norm = self._normalize_family_name(family_raw)
            if family_norm and family_norm not in self._family_display_names:
                self._family_display_names[family_norm] = family_raw

            text = str(rec.get("text", "") or "")
            if "học vụ" in family_norm and self._contains_graduation_scale_signal(text):
                self._graduation_scale_indices.append(idx)
            m_art = re.search(r'(?:^|\n)\s*Điều\s+(\d+)[\s:.]', text, re.IGNORECASE)
            if m_art:
                article = m_art.group(1)
            elif rec.get("article") is not None:
                article = str(rec.get("article"))
            else:
                article = ""

            if article:
                self._article_to_indices.setdefault(article, []).append(idx)
                if family_norm:
                    self._family_article_to_indices.setdefault((family_norm, article), []).append(idx)
                    self._article_families.setdefault(article, set()).add(family_norm)

        for idx, rec in enumerate(self._records):
            if not isinstance(rec, dict):
                continue
            text = str(rec.get("text", "") or "")
            refs = self._extract_inline_references(text)
            if not refs:
                continue

            family_raw = str(
                rec.get("doc_family")
                or rec.get("source_title")
                or rec.get("document_name")
                or ""
            )
            family_norm = self._normalize_family_name(family_raw)
            targets: list[int] = []
            for ref in refs:
                article = str(ref.get("article") or "").strip()
                if not article:
                    continue
                same_family_targets = self._family_article_to_indices.get((family_norm, article), [])
                global_targets = self._article_to_indices.get(article, [])
                candidate_targets = same_family_targets if same_family_targets else global_targets
                for t_idx in candidate_targets[:6]:
                    if t_idx == idx or t_idx in targets:
                        continue
                    targets.append(t_idx)
                    if len(targets) >= 8:
                        break
                if len(targets) >= 8:
                    break
            if targets:
                self._xref_links[idx] = targets

    def _expand_with_cross_document(self, query: str, results: list[dict], max_extra: int = 6) -> list[dict]:
        """Expand retrieved contexts with cross-referenced chunks."""
        if not results or not self._record_key_to_idx or not self._records:
            return results

        enriched = list(results)
        seen = {self._make_record_key(r) for r in results if isinstance(r, dict)}
        extras: list[dict] = []

        for r in results[:8]:
            if not isinstance(r, dict):
                continue
            src_idx = self._record_key_to_idx.get(self._make_record_key(r))
            if src_idx is None:
                continue
            for t_idx in self._xref_links.get(src_idx, []):
                if not (0 <= t_idx < len(self._records)):
                    continue
                candidate = self._records[t_idx]
                if not isinstance(candidate, dict):
                    continue
                c_key = self._make_record_key(candidate)
                if c_key in seen:
                    continue
                seen.add(c_key)
                extras.append(
                    {
                        **candidate,
                        "retrieval_source": "xref",
                        "rerank_score": float(r.get("rerank_score", r.get("final_score", 0.0))) - 0.03,
                        "final_score": float(r.get("final_score", r.get("rerank_score", 0.0))) - 0.03,
                    }
                )
                if len(extras) >= max_extra:
                    break
            if len(extras) >= max_extra:
                break

        # Nếu query nêu Điều cụ thể, bổ sung thêm đại diện từ các văn bản liên quan.
        qmeta = self._parse_query_metadata(query)
        q_family = self._infer_query_doc_family(query)
        if qmeta.get("article") is not None:
            article = str(qmeta["article"])
            family_set = sorted(self._article_families.get(article, set()))
            for fam in family_set:
                if q_family and self._normalize_family_name(q_family) != fam:
                    continue
                fam_targets = self._family_article_to_indices.get((fam, article), [])
                for t_idx in fam_targets[:1]:
                    if not (0 <= t_idx < len(self._records)):
                        continue
                    candidate = self._records[t_idx]
                    if not isinstance(candidate, dict):
                        continue
                    c_key = self._make_record_key(candidate)
                    if c_key in seen:
                        continue
                    seen.add(c_key)
                    extras.append(
                        {
                            **candidate,
                            "retrieval_source": "xref_article",
                            "rerank_score": -0.02,
                            "final_score": -0.02,
                        }
                    )
                    if len(extras) >= max_extra:
                        break
                if len(extras) >= max_extra:
                    break

        if extras:
            enriched.extend(extras)
        return enriched

    def build_database(self) -> bool:
        """Build vector database and BM25 indexes from CSV and PDF files."""
        backend_name = "Qdrant" if self._is_qdrant_backend() else "FAISS"
        print(f"🚀 Bắt đầu xây dựng cơ sở dữ liệu {backend_name} + BM25...")
        local_records: list[dict] = []

        # Load CSV
        if os.path.exists(self.config.csv_file):
            try:
                df = pd.read_csv(self.config.csv_file, encoding="utf-8-sig")
                if "question" in df.columns and "answer" in df.columns:
                    for _, row in df.iterrows():
                        q = preprocess_text(str(row["question"]))
                        a = preprocess_text(str(row["answer"]))
                        # CSV: dùng hierarchical chunking (256 tokens) để giữ cấu trúc Điều/Khoản
                        chunks = chunk_text_hierarchical(a, self.config.csv_chunk_max_tokens)
                        if not chunks:
                            # Fallback về chunk_text nếu hierarchical trả rỗng
                            chunks = chunk_text(a, self.config.csv_chunk_max_tokens, self.config.chunk_overlap_tokens)
                        if not chunks:
                            chunks = [a] if a else []
                        for chunk in chunks:
                            local_records.append({
                                "question": q,
                                "text": chunk,
                                "source": os.path.basename(self.config.csv_file),
                                "document_name": self.config.csv_document_name,
                            })
                    print(f"✔️ Đã đọc {len(local_records)} QA từ CSV.")
                else:
                    print("⚠️ CSV không có cột 'question' và 'answer'!")
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc CSV: {e}")

        # Load PDF with Structure-aware Chunking
        if os.path.exists(self.config.pdf_file):
            try:
                # Lấy raw text (giữ \n) để nhận dạng cấu trúc
                pdf_text = extract_pdf_text(self.config.pdf_file)
                doc_ranges = infer_pdf_doc_ranges(self.config.pdf_file)
                if doc_ranges:
                    print("🗂️ Phát hiện các vùng văn bản chính từ mục lục:")
                    for dr in doc_ranges:
                        print(
                            f"   - {dr.get('doc_family')} | trang {dr.get('start_page')}–{dr.get('end_page')}"
                        )
                # Sử dụng structure-aware chunking cho PDF quy chế (chunk 1024 token)
                pdf_records = chunk_pdf_by_structure(
                    pdf_text, 
                    self.config.chunk_max_tokens,
                    document_name=self.config.pdf_document_name,
                    doc_ranges=doc_ranges,
                )
                # Cập nhật source với tên file thực
                for rec in pdf_records:
                    rec["source"] = os.path.basename(self.config.pdf_file)
                local_records.extend(pdf_records)
                print(f"✅ Đã đọc {len(pdf_records)} đoạn từ PDF (structure-aware chunking).")
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc PDF: {e}")

        # Add metadata (chỉ cho records chưa có metadata từ structure chunker)
        for i in range(len(local_records)):
            record = local_records[i]
            if not isinstance(record, dict):
                record = {"question": "", "text": str(record)}
            question = preprocess_text(str(record.get("question", "") or ""))
            text = preprocess_text(str(record.get("text", "") or ""))
            source = str(record.get("source") or "")
            
            # Chỉ extract metadata nếu record thực sự chưa có metadata cấu trúc.
            # Với records từ structure chunker, luôn ưu tiên giữ metadata gốc.
            has_structured_meta = (
                bool(record.get("structure_type"))
                or bool(record.get("header_span"))
                or record.get("article") is not None
                or record.get("chapter") is not None
                or bool(record.get("section"))
                or record.get("clause") is not None
            )
            if not has_structured_meta:
                meta = extract_metadata(f"{question}\n{text}")
            else:
                # Giữ nguyên metadata từ structure chunker
                meta = {
                    "chapter": record.get("chapter"),
                    "section": record.get("section"),
                    "article": record.get("article"),
                    "clause": record.get("clause"),
                }
            preserved = record.copy() if isinstance(record, dict) else {}
            local_records[i] = {
                **preserved,
                "question": question,
                "text": text,
                "source": source,
                **meta,
            }

        print(f"📚 Tổng số đoạn sau xử lý: {len(local_records)}")
        if not local_records:
            print("❌ Không có dữ liệu để xây dựng vector database!")
            return False

        try:
            def _mem_mb() -> str:
                if PSUTIL_AVAILABLE:
                    return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.0f}MB"
                return "N/A"

            print(f"💾 RAM trước build: {_mem_mb()}")
            self._load_embedder()

            texts = [self._record_index_text(r) for r in local_records]
            total = len(texts)
            BATCH_SIZE = 500

            # --- Batch encoding để giảm RAM peak ---
            all_embeddings: list[np.ndarray] = []
            print(f"🔄 Đang encode {total} văn bản (batch size: {BATCH_SIZE})...")
            for start in range(0, total, BATCH_SIZE):
                batch_texts = texts[start:start + BATCH_SIZE]
                batch_num = start // BATCH_SIZE + 1
                total_batches = (total - 1) // BATCH_SIZE + 1
                print(f"   📦 Batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                batch_emb = self._embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                all_embeddings.append(batch_emb)
                del batch_emb

            embeddings = np.vstack(all_embeddings)
            del all_embeddings
            gc.collect()

            # --- Giải phóng embedder ngay sau encode (~700MB RAM) ---
            print("🧹 Giải phóng embedding model khỏi RAM...")
            del self._embedder
            self._embedder = None
            gc.collect()
            print(f"💾 RAM sau giải phóng embedder: {_mem_mb()}")

            # Build BM25
            print("📊 Đang xây dựng BM25 index...")
            self._tokenized_corpus = [text.lower().split() for text in texts]
            self._bm25_index = BM25Okapi(self._tokenized_corpus)
            with open(self.config.bm25_index_path, "wb") as f:
                pickle.dump({"bm25": self._bm25_index, "tokenized_corpus": self._tokenized_corpus}, f)

            del texts
            gc.collect()

            # Build vector index
            embeddings = embeddings.astype("float32")
            if self._is_qdrant_backend():
                self._ensure_qdrant_collection(
                    vector_size=int(embeddings.shape[1]),
                    recreate=self.config.qdrant_recreate_on_build,
                )
                client = self._init_qdrant_client()
                if client is None:
                    raise RuntimeError("Không thể kết nối Qdrant client")
                upsert_batch = 256
                total_points = len(local_records)
                print(
                    f"🛰️ Đang upload vectors lên Qdrant collection '{self.config.qdrant_collection}' "
                    f"({total_points} points, batch={upsert_batch})..."
                )
                for start in range(0, total_points, upsert_batch):
                    end = min(start + upsert_batch, total_points)
                    points = [
                        PointStruct(
                            id=int(i),
                            vector=embeddings[i].tolist(),
                            payload=self._make_qdrant_payload(local_records[i], i),
                        )
                        for i in range(start, end)
                    ]
                    client.upsert(
                        collection_name=self.config.qdrant_collection,
                        points=points,
                        wait=True,
                    )
                self._index = self._qdrant_client
            else:
                if not FAISS_AVAILABLE:
                    raise RuntimeError("faiss chưa sẵn sàng. Hãy cài faiss-cpu hoặc chuyển VECTOR_DB_TYPE=qdrant")
                faiss.normalize_L2(embeddings)
                self._index = faiss.IndexFlatIP(embeddings.shape[1])
                self._index.add(embeddings)
                faiss.write_index(self._index, self.config.vector_index_path)

            del embeddings
            gc.collect()

            with open(self.config.vector_texts_path, "wb") as f:
                pickle.dump(local_records, f)

            self._records = local_records
            self._build_cross_document_mapping()
            print(f"✅ Database built thành công. RAM hiện tại: {_mem_mb()}")
            print("💡 Embedder sẽ tự động load lại khi có query đầu tiên.")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi xây dựng database: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_database(self) -> tuple[list[dict], Any]:
        """Load existing vector database and BM25 indexes."""
        backend_name = "Qdrant" if self._is_qdrant_backend() else "FAISS"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._is_qdrant_backend():
                    if not os.path.exists(self.config.vector_texts_path):
                        print("📦 Không tìm thấy vector_texts.pkl, tiến hành rebuild...")
                        if not self.build_database():
                            return [], None
                        continue

                    with open(self.config.vector_texts_path, "rb") as f:
                        self._records = pickle.load(f)

                    client = self._init_qdrant_client()
                    if client is None:
                        raise RuntimeError("Không thể khởi tạo Qdrant client")
                    if not client.collection_exists(self.config.qdrant_collection):
                        print(
                            f"📦 Không tìm thấy Qdrant collection '{self.config.qdrant_collection}', "
                            "tiến hành rebuild..."
                        )
                        if not self.build_database():
                            return [], None
                        continue
                    try:
                        count_result = client.count(
                            collection_name=self.config.qdrant_collection,
                            exact=True,
                        )
                        point_count = int(getattr(count_result, "count", 0))
                        if self._records and point_count <= 0:
                            print("⚠️ Qdrant collection đang rỗng, tiến hành rebuild...")
                            if not self.build_database():
                                return [], None
                            continue
                        if self._records and point_count < len(self._records):
                            print(
                                f"⚠️ Qdrant có {point_count} points < {len(self._records)} records. "
                                "Nên rebuild để đồng bộ dữ liệu."
                            )
                    except Exception:
                        pass
                    self._index = self._qdrant_client
                else:
                    if not FAISS_AVAILABLE:
                        raise RuntimeError("faiss chưa sẵn sàng. Hãy cài faiss-cpu hoặc chuyển VECTOR_DB_TYPE=qdrant")
                    if not os.path.exists(self.config.vector_index_path) or not os.path.exists(self.config.vector_texts_path):
                        print("📦 Không tìm thấy file FAISS, tiến hành rebuild...")
                        if not self.build_database():
                            return [], None
                        continue

                    self._index = faiss.read_index(self.config.vector_index_path)
                    try:
                        expected_metric = faiss.METRIC_INNER_PRODUCT
                        current_metric = getattr(self._index, "metric_type", None)
                        if current_metric is not None and current_metric != expected_metric:
                            print("⚠️ FAISS index đang dùng metric cũ. Tiến hành rebuild...")
                            if not self.build_database():
                                return [], None
                            continue
                    except Exception:
                        pass

                    with open(self.config.vector_texts_path, "rb") as f:
                        self._records = pickle.load(f)

                # Add metadata if missing
                try:
                    for i in range(len(self._records)):
                        rec = self._records[i]
                        if not isinstance(rec, dict):
                            continue
                        if rec.get("article") is None and rec.get("section") is None and rec.get("chapter") is None and rec.get("clause") is None:
                            meta = extract_metadata(f"{rec.get('question','')}\n{rec.get('text','')}")
                            rec.update(meta)
                except Exception:
                    pass

                # Ensure critical metadata fields exist for citation formatting.
                try:
                    for i in range(len(self._records)):
                        rec = self._records[i]
                        if not isinstance(rec, dict):
                            continue
                        if not rec.get("document_name"):
                            source_name = str(rec.get("source") or "").lower()
                            rec["document_name"] = (
                                self.config.pdf_document_name
                                if source_name.endswith(".pdf")
                                else self.config.csv_document_name
                            )
                        if not rec.get("structure_type"):
                            if rec.get("article") is not None:
                                rec["structure_type"] = "article"
                            elif rec.get("section"):
                                rec["structure_type"] = "section"
                            elif rec.get("header_span"):
                                rec["structure_type"] = "title"
                            else:
                                rec["structure_type"] = "unknown"
                        if not rec.get("source_title"):
                            rec["source_title"] = (
                                str(rec.get("section_title") or "").strip()
                                or str(rec.get("header_span") or "").strip()
                                or str(rec.get("document_name") or "").strip()
                            )
                        if not rec.get("doc_family"):
                            rec["doc_family"] = str(rec.get("source_title") or rec.get("document_name") or "Văn bản liên quan")
                        if rec.get("page_start") is None:
                            rec["page_start"] = None
                        if rec.get("page_end") is None:
                            rec["page_end"] = rec.get("page_start")
                except Exception:
                    pass

                # Load BM25
                if os.path.exists(self.config.bm25_index_path):
                    with open(self.config.bm25_index_path, "rb") as f:
                        bm25_data = pickle.load(f)
                        self._bm25_index = bm25_data["bm25"]
                        self._tokenized_corpus = bm25_data["tokenized_corpus"]
                else:
                    print("⚠️ Không tìm thấy BM25 index, cần rebuild database.")

                self._build_cross_document_mapping()
                print(f"✅ Đã load database {backend_name}. Số lượng: {len(self._records)} records")
                return self._records, self._index
            except Exception as e:
                print(f"⚠️ Lỗi load {backend_name} (lần {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print("🔄 Thử rebuild database...")
                    if not self.build_database():
                        return [], None
                else:
                    return [], None
        return [], None

    # ==========================================================================
    # Retrieval
    # ==========================================================================

    def _encode_query_cached(self, query: str) -> np.ndarray:
        """Cache query embeddings để tránh encode lại query giống nhau."""
        if query in self._query_embedding_cache:
            self._query_embedding_cache.move_to_end(query)
            return self._query_embedding_cache[query]
        
        # Lazy-load embedder nếu đã bị unload sau build_database()
        self._load_embedder()
        
        q_vec = self._embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        if not self._is_qdrant_backend() and FAISS_AVAILABLE:
            faiss.normalize_L2(q_vec)
        
        self._query_embedding_cache[query] = q_vec
        self._query_embedding_cache.move_to_end(query)
        while len(self._query_embedding_cache) > self._query_cache_maxsize:
            self._query_embedding_cache.popitem(last=False)
        
        return q_vec

    def _reciprocal_rank_fusion(self, dense_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
        """Combine dense and sparse results using RRF."""
        scores: dict[str, dict] = {}
        for rank, result in enumerate(dense_results):
            key_payload = f"{result.get('source','')}|{result.get('question','')}|{result.get('text','')}"
            doc_key = hashlib.sha1(key_payload.encode("utf-8", errors="ignore")).hexdigest()
            scores.setdefault(doc_key, {"data": result, "score": 0})
            scores[doc_key]["score"] += 1 / (k + rank + 1)
        for rank, result in enumerate(bm25_results):
            key_payload = f"{result.get('source','')}|{result.get('question','')}|{result.get('text','')}"
            doc_key = hashlib.sha1(key_payload.encode("utf-8", errors="ignore")).hexdigest()
            scores.setdefault(doc_key, {"data": result, "score": 0})
            scores[doc_key]["score"] += 1 / (k + rank + 1)
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["data"] for item in sorted_results]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Hybrid retrieval using vector DB + BM25 with RRF fusion."""
        top_k = top_k or self.config.top_k
        try:
            if self._index is None or not self._records:
                return []

            # Dense retrieval (FAISS/Qdrant) - sử dụng cached embedding
            q_vec = self._encode_query_cached(query)
            dense_results: list[dict] = []
            dense_limit = min(top_k * 2, len(self._records))

            if self._is_qdrant_backend():
                client = self._init_qdrant_client()
                if client is None:
                    return []
                response = client.query_points(
                    collection_name=self.config.qdrant_collection,
                    query=q_vec[0].tolist(),
                    limit=dense_limit,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in getattr(response, "points", []):
                    idx: Optional[int] = None
                    raw_id = getattr(point, "id", None)
                    if isinstance(raw_id, int):
                        idx = raw_id
                    elif isinstance(raw_id, str) and raw_id.isdigit():
                        idx = int(raw_id)

                    payload = getattr(point, "payload", None) or {}
                    if idx is None:
                        payload_idx = payload.get("record_idx")
                        if isinstance(payload_idx, int):
                            idx = payload_idx
                        elif isinstance(payload_idx, str) and payload_idx.isdigit():
                            idx = int(payload_idx)

                    if idx is None or not (0 <= idx < len(self._records)):
                        continue
                    dense_results.append({
                        **self._records[idx],
                        "retrieval_source": "dense",
                        "dense_score": float(getattr(point, "score", 0.0)),
                    })
            else:
                distances, indices = self._index.search(q_vec, dense_limit)
                for pos, idx in enumerate(indices[0]):
                    if 0 <= int(idx) < len(self._records):
                        dense_results.append({
                            **self._records[int(idx)],
                            "retrieval_source": "dense",
                            "dense_score": float(distances[0][pos]),
                        })

            # Sparse retrieval (BM25)
            bm25_results: list[dict] = []
            if self._bm25_index is not None and self._tokenized_corpus:
                query_tokens = preprocess_text(query).lower().split()
                bm25_scores = self._bm25_index.get_scores(query_tokens)
                top_bm25 = np.argsort(bm25_scores)[::-1][: min(top_k * 2, len(self._records))]
                for idx in top_bm25:
                    if bm25_scores[idx] <= 0:
                        continue
                    bm25_results.append({
                        **self._records[int(idx)],
                        "retrieval_source": "bm25",
                        "bm25_score": float(bm25_scores[idx]),
                    })

            # Fusion
            fused_results = self._reciprocal_rank_fusion(dense_results, bm25_results) if bm25_results else dense_results
            return fused_results[:top_k]
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ==========================================================================
    # Reranking & Boosting
    # ==========================================================================

    def rerank_results(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    
        if not results:
            return results
        
        # Stage 1: BM25 fast filtering (giữ top 10 nếu đầu vào quá nhiều)
        stage1_results = results
        if len(results) > 10 and self._bm25_index is not None and self._tokenized_corpus:
            try:
                query_tokens = preprocess_text(query).lower().split()
                
                # Tính BM25 score cho từng result dựa trên text match
                for r in results:
                    doc_text = f"{r.get('question', '')} {r.get('text', '')}".lower()
                    doc_tokens = doc_text.split()
                    # Quick relevance score dựa trên token overlap
                    overlap = len(set(query_tokens) & set(doc_tokens))
                    r["_bm25_quick_score"] = overlap / max(len(query_tokens), 1)
                
                # Sort by quick score và giữ top 10
                results_sorted = sorted(results, key=lambda x: x.get("_bm25_quick_score", 0), reverse=True)
                stage1_results = results_sorted[:10]
            except Exception:
                stage1_results = results[:10]
        
        # Stage 2: Cross-encoder deep reranking (chỉ trên 10 candidates)
        reranker_model = self._load_reranker()
        if reranker_model is None:
            return stage1_results[:top_k]
        try:
            pairs = []
            for r in stage1_results:
                doc = f"{r.get('question', '')}\n{r.get('text', '')}".strip()
                pairs.append([query, doc])
            scores = reranker_model.predict(pairs, show_progress_bar=False)
            for i, result in enumerate(stage1_results):
                result["rerank_score"] = float(scores[i])
            reranked = sorted(stage1_results, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
            return stage1_results[:top_k]

    def _parse_query_metadata(self, query: str) -> dict:
        """Parse metadata references from query."""
        q = (query or "").lower()
        wants_article = "điều" in q
        wants_clause = "khoản" in q
        wants_section = "mục" in q
        wants_chapter = "chương" in q

        article_num = None
        clause_num = None
        section_id = None
        chapter_id = None

        m = re.search(r"(?:^|\b)điều\s*(\d+)\b", q)
        if m:
            try:
                article_num = int(m.group(1))
            except Exception:
                article_num = None

        m = re.search(r"(?:^|\b)khoản\s*(\d+)\b", q)
        if m:
            try:
                clause_num = int(m.group(1))
            except Exception:
                clause_num = None

        m = re.search(r"(?:^|\b)mục\s*([ivxlcdm]+|\d+)\b", q)
        if m:
            section_id = m.group(1).upper()

        m = re.search(r"(?:^|\b)chương\s*([ivxlcdm]+|\d+)\b", q)
        if m:
            chapter_id = m.group(1).upper()

        return {
            "wants_article": wants_article,
            "wants_clause": wants_clause,
            "wants_section": wants_section,
            "wants_chapter": wants_chapter,
            "article": article_num,
            "clause": clause_num,
            "section": section_id,
            "chapter": chapter_id,
        }

    def _infer_query_doc_families(self, query: str) -> list[str]:
        """Infer one or more relevant handbook families from query intent."""
        q = (query or "").lower()
        families: list[str] = []

        def _add(name: str) -> None:
            if name not in families:
                families.append(name)

        if any(k in q for k in ["học bổng", "hbkkht", "khuyến khích học tập"]):
            _add("Quy định học bổng")
        if any(k in q for k in ["miễn giảm học phí", "miễn, giảm học phí", "trợ cấp", "chính sách"]):
            _add("Chế độ, chính sách sinh viên")
        if any(k in q for k in ["công tác sinh viên", "rèn luyện", "kỷ luật sinh viên", "khiển trách", "cảnh cáo"]):
            _add("Quy chế Công tác sinh viên")
        if any(k in q for k in ["học vụ", "đào tạo", "học phần", "tín chỉ", "bảo lưu", "tốt nghiệp", "gpa", "xếp loại"]):
            _add("Quy chế Học vụ")

        return families

    def _infer_query_doc_family(self, query: str) -> str | None:
        """
        Infer a single dominant family.
        - Return explicit family if user gọi đích danh.
        - Return None nếu query chạm nhiều family để tránh boost lệch 1 nguồn.
        """
        q = (query or "").lower()
        if "quy chế học vụ" in q:
            return "Quy chế Học vụ"
        if "quy chế công tác sinh viên" in q:
            return "Quy chế Công tác sinh viên"
        if "quy định học bổng" in q:
            return "Quy định học bổng"
        if "chế độ, chính sách sinh viên" in q or "chế độ chính sách đối với sinh viên" in q:
            return "Chế độ, chính sách sinh viên"

        families = self._infer_query_doc_families(query)
        return families[0] if len(families) == 1 else None

    def _is_graduation_scale_query(self, query: str) -> bool:
        q = (query or "").lower()
        has_grad_topic = any(
            k in q
            for k in ["tốt nghiệp", "hạng tốt nghiệp", "xếp loại tốt nghiệp", "xếp loại học lực", "gpa"]
        )
        has_scale_signal = any(
            k in q
            for k in ["thang điểm", "bao nhiêu", "xuất sắc", "giỏi", "khá", "trung bình", "3.6", "3,6"]
        )
        return has_grad_topic and has_scale_signal

    def _contains_graduation_scale_signal(self, text: str) -> bool:
        t = (text or "").lower()
        has_table_head = ("điểm chữ" in t and "điểm số hệ 4" in t) or ("xếp loại học lực" in t)
        has_range_row = bool(
            re.search(r"3\s*[,.]\s*60\s*(?:đến|-)\s*4\s*[,.]\s*00", t, re.IGNORECASE)
        )
        has_labels = ("xuất sắc" in t and "giỏi" in t and "khá" in t)
        return has_table_head or (has_range_row and has_labels)

    def _inject_required_context(self, query: str, results: list[dict], max_extra: int = 4) -> list[dict]:
        """Force-include must-have chunks for critical query types (e.g. graduation scale)."""
        if not results or not self._records:
            return results
        if not self._is_graduation_scale_query(query):
            return results

        existing_blob = "\n".join(str(r.get("text", "") or "") for r in results[:12] if isinstance(r, dict))
        if self._contains_graduation_scale_signal(existing_blob):
            return results

        seen = {self._make_record_key(r) for r in results if isinstance(r, dict)}
        extras: list[dict] = []
        base_score = float(results[min(2, len(results) - 1)].get("final_score", results[min(2, len(results) - 1)].get("rerank_score", 0.0)))

        candidate_indices = self._graduation_scale_indices or list(range(len(self._records)))
        for idx in candidate_indices:
            if not (0 <= idx < len(self._records)):
                continue
            rec = self._records[idx]
            if not isinstance(rec, dict):
                continue
            fam = str(rec.get("doc_family") or "").lower()
            if "học vụ" not in fam:
                continue
            text = str(rec.get("text", "") or "")
            if not self._contains_graduation_scale_signal(text):
                continue
            k = self._make_record_key(rec)
            if k in seen:
                continue
            seen.add(k)
            extras.append(
                {
                    **rec,
                    "retrieval_source": "required_context",
                    "rerank_score": base_score - 0.015,
                    "final_score": base_score - 0.015,
                }
            )
            if len(extras) >= max_extra:
                break

        if not extras:
            return results
        return sorted(results + extras, key=lambda x: x.get("final_score", x.get("rerank_score", 0.0)), reverse=True)

    def _remove_false_missing_context_claim(self, answer: str, query: str, top_records: list[dict]) -> str:
        """Drop 'không có trong ngữ cảnh' claims when evidence exists in retrieved context."""
        if not answer:
            return answer
        if not self._is_graduation_scale_query(query):
            return answer

        context_blob = "\n".join(str(r.get("text", "") or "") for r in top_records if isinstance(r, dict))
        if not self._contains_graduation_scale_signal(context_blob):
            return answer

        cleaned = re.sub(
            r'[^.\n]{0,160}không\s+(?:được\s+cung\s+cấp|có)\s+trong\s+ngữ\s+cảnh[^.\n]*\.?',
            '',
            answer,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r'\*?\(?\s*Theo văn bản liên quan(?:[^)\n]{0,80})\)?\*?',
            '',
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned

    def _collect_constraint_units(self, raw_text: str) -> list[str]:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return []
        units = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        units.extend([s.strip() for s in re.split(r'(?<=[.!?;])\s+', raw_text) if len(s.strip()) >= 20])
        seen: set[str] = set()
        results: list[str] = []
        for u in units:
            norm = re.sub(r"\s+", " ", u).strip()
            norm_key = norm.lower()
            if not norm or norm_key in seen:
                continue
            seen.add(norm_key)
            results.append(norm)
        return results

    def _build_constraint_report(self, query: str, records: list[dict]) -> dict:
        query_l = (query or "").lower()
        corpus_parts = []
        for r in records:
            if isinstance(r, dict):
                corpus_parts.append(str(r.get("text", "") or ""))
        corpus = "\n".join(corpus_parts)
        corpus_l = corpus.lower()

        # Chỉ đánh dấu "câu hỏi ra quyết định cá nhân" khi có tín hiệu thật sự.
        # Tránh bắt người dùng nhập dữ liệu cá nhân cho câu hỏi thông tin chung.
        decision_query = any(
            k in query_l
            for k in [
                "đủ điều kiện",
                "có được",
                "được không",
                "trường hợp của em",
                "trường hợp của tôi",
                "tôi có",
                "em có",
                "có bị",
            ]
        )

        rules = [
            {
                "id": "hb_min_credits",
                "title": "Tối thiểu 15 tín chỉ đăng ký lần đầu trong kỳ xét học bổng.",
                "group": "apply",
                "triggers": ["học bổng", "hbkkht"],
                "evidence_any": ["15 tín chỉ"],
                "required_input_hint": "Số tín chỉ đăng ký lần đầu trong kỳ xét.",
                "input_match": r'\b\d+\s*tín\s*chỉ\b',
            },
            {
                "id": "hb_score_floor",
                "title": "Điểm xét học bổng lần thi thứ nhất, không có điểm dưới 5,5 (thang 10).",
                "group": "exclude",
                "triggers": ["học bổng", "hbkkht"],
                "evidence_any": ["5,5", "lần thứ nhất", "không có điểm thi dưới"],
                "required_input_hint": "Điểm từng học phần ở lần thi thứ nhất (thang 10).",
                "input_match": r'(điểm|5,5|thang\s*10)',
            },
            {
                "id": "grad_rank_downgrade_f",
                "title": "Hạng tốt nghiệp bị hạ 1 mức nếu tín chỉ học lại (điểm F) vượt quá 5% chương trình.",
                "group": "exclude",
                "triggers": ["tốt nghiệp", "hạng tốt nghiệp", "điều 43"],
                "evidence_any": ["điểm f", "5%", "vượt quá"],
                "required_input_hint": "Tỷ lệ tín chỉ học lại (điểm F) trên tổng chương trình.",
                "input_match": r'(điểm\s*f|học\s*lại|5%)',
            },
            {
                "id": "grad_rank_downgrade_discipline",
                "title": "Hạng tốt nghiệp bị hạ 1 mức nếu bị kỷ luật từ mức Cảnh cáo trở lên.",
                "group": "exclude",
                "triggers": ["tốt nghiệp", "hạng tốt nghiệp", "điều 43"],
                "evidence_any": ["cảnh cáo", "kỷ luật"],
                "required_input_hint": "Thông tin kỷ luật trong thời gian học.",
                "input_match": r'(kỷ\s*luật|cảnh\s*cáo)',
            },
        ]

        applicable_rules: list[str] = []
        exclusion_rules: list[str] = []
        critical_missing: list[str] = []
        missing_inputs: list[str] = []

        for rule in rules:
            # Kích hoạt rule dựa trên query để tránh nhiễu chéo từ context retrieval.
            rule_active = any(t in query_l for t in rule["triggers"])
            if not rule_active:
                continue
            rule_found = any(ev in corpus_l for ev in rule["evidence_any"])
            if rule_found:
                if rule["group"] == "exclude":
                    exclusion_rules.append(rule["title"])
                else:
                    applicable_rules.append(rule["title"])
            else:
                critical_missing.append(rule["title"])

            if decision_query and not re.search(str(rule["input_match"]), query_l, re.IGNORECASE):
                missing_inputs.append(str(rule["required_input_hint"]))

        # Bổ sung các câu điều kiện tổng quát từ context để tránh bỏ sót chi tiết.
        generic_apply_kw = ["điều kiện", "phải", "tối thiểu", "được xét", "đạt"]
        generic_exclude_kw = ["không", "trừ", "ngoại lệ", "bị giảm", "bị hạ", "vượt quá"]
        for line in self._collect_constraint_units(corpus):
            ll = line.lower()
            if len(applicable_rules) < 8 and any(k in ll for k in generic_apply_kw):
                if line not in applicable_rules:
                    applicable_rules.append(line)
            if len(exclusion_rules) < 8 and any(k in ll for k in generic_exclude_kw):
                if line not in exclusion_rules:
                    exclusion_rules.append(line)

        def _dedupe_keep_order(items: list[str], limit: int = 8) -> list[str]:
            out: list[str] = []
            seen_local: set[str] = set()
            for item in items:
                key = re.sub(r"\s+", " ", item.strip().lower())
                if not key or key in seen_local:
                    continue
                seen_local.add(key)
                out.append(item.strip())
                if len(out) >= limit:
                    break
            return out

        applicable_rules = _dedupe_keep_order(applicable_rules)
        exclusion_rules = _dedupe_keep_order(exclusion_rules)
        critical_missing = _dedupe_keep_order(critical_missing, limit=5)
        missing_inputs = _dedupe_keep_order(missing_inputs, limit=5)

        return {
            "decision_query": decision_query,
            "applicable_rules": applicable_rules,
            "exclusion_rules": exclusion_rules,
            "critical_missing": critical_missing,
            "missing_inputs": missing_inputs,
        }

    def _metadata_boost(self, query: str, record: dict) -> float:
        """Calculate metadata-based score boost."""
        qmeta = self._parse_query_metadata(query)
        q_family = self._infer_query_doc_family(query)
        q_families = [f.lower() for f in self._infer_query_doc_families(query)]
        boost = 0.0
        cfg = self.config

        rec_article = record.get("article")
        rec_clause = record.get("clause")
        rec_section = record.get("section")
        rec_chapter = record.get("chapter")
        rec_family = str(record.get("doc_family") or "")

        if qmeta["article"] is not None and rec_article == qmeta["article"]:
            boost += cfg.meta_boost_article_match
        elif qmeta["wants_article"] and rec_article is not None:
            boost += cfg.meta_boost_article_present

        if qmeta["clause"] is not None and rec_clause == qmeta["clause"]:
            boost += cfg.meta_boost_clause_match
        elif qmeta["wants_clause"] and rec_clause is not None:
            boost += cfg.meta_boost_clause_present

        if qmeta["section"] is not None and rec_section == qmeta["section"]:
            boost += cfg.meta_boost_section_match
        elif qmeta["wants_section"] and rec_section:
            boost += cfg.meta_boost_section_present

        if qmeta["chapter"] is not None and rec_chapter == qmeta["chapter"]:
            boost += cfg.meta_boost_chapter_match
        elif qmeta["wants_chapter"] and rec_chapter:
            boost += cfg.meta_boost_chapter_present

        if rec_family and q_families:
            if rec_family.lower() in q_families:
                boost += 0.12 if len(q_families) > 1 else 0.18
        elif q_family and rec_family and q_family.lower() == rec_family.lower():
            boost += 0.18

        return min(cfg.meta_boost_max, max(0.0, boost))

    def apply_metadata_boost(self, query: str, results: list[dict]) -> list[dict]:
        """Apply metadata boosting to results."""
        boosted = []
        for r in results:
            base = float(r.get("rerank_score", 0.0))
            b = self._metadata_boost(query, r)
            r["meta_boost"] = b
            r["final_score"] = base + b
            boosted.append(r)
        return sorted(boosted, key=lambda x: x.get("final_score", x.get("rerank_score", 0.0)), reverse=True)

    def calculate_confidence(self, results: list[dict]) -> tuple[float, str]:
        """Calculate retrieval confidence score."""
        if not results:
            return 0.0, "very_low"
        scores = [r.get("final_score", r.get("rerank_score", 0.5)) for r in results[:3]]
        avg_score = sum(scores) / len(scores) if scores else 0
        score_gap = scores[0] - scores[1] if len(scores) > 1 else 0
        confidence = min(1.0, max(0.0, avg_score * 0.7 + score_gap * 0.3))
        if confidence >= 0.8:
            label = "high"
        elif confidence >= 0.6:
            label = "medium"
        elif confidence >= 0.4:
            label = "low"
        else:
            label = "very_low"
        return confidence, label

    # ==========================================================================
    # Gemini API
    # ==========================================================================

    def _get_gemini_model(self):
        """Get or create Gemini model instance."""
        if self._gemini_model is not None:
            return self._gemini_model
        
        # Use SecretsManager if available for secure API key handling
        GEMINI_API_KEY = None
        if SECRETS_MANAGER_AVAILABLE:
            try:
                secrets = get_secrets_manager()
                GEMINI_API_KEY = secrets.get_gemini_key(validate=True)
            except Exception as e:
                # Fallback to environment variable
                print(f"⚠️ SecretsManager failed, using env: {e}")
        
        # Fallback to direct environment variable
        if not GEMINI_API_KEY:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        if not GEMINI_API_KEY:
            raise ValueError("❌ Thiếu GEMINI_API_KEY trong .env/.env.txt")
        
        genai.configure(api_key=GEMINI_API_KEY, transport="rest")
        self._gemini_model = genai.GenerativeModel(self.config.gemini_model)
        return self._gemini_model

    def configure_gemini(self) -> bool:
        """Test and configure Gemini API connection."""
        max_retries = 2
        base_delay = 5
        for attempt in range(max_retries):
            try:
                model = self._get_gemini_model()
                test = model.generate_content("Test connection - Xin chào!")
                if hasattr(test, "text") and test.text:
                    self._gemini_ready = True
                    self.clear_answer_cache()
                    return True
            except Exception as e:
                if "429" in str(e):
                    time.sleep(base_delay * (attempt + 1))
                else:
                    break
        self._gemini_ready = False
        self.clear_answer_cache()
        return False

    def disconnect_gemini(self) -> None:
        """Disconnect Gemini API."""
        self._gemini_ready = False
        self.clear_answer_cache()

    def _ask_gemini(self, prompt: str, temperature: float = 0.3, max_output_tokens: int = 8192) -> str:
        """Send prompt to Gemini and get response."""
        if not self._gemini_ready:
            return "❌ Gemini API chưa sẵn sàng. Nhấn 'Kết nối API' để thử lại."

        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        if time_since_last_call < self.config.min_request_interval:
            time.sleep(self.config.min_request_interval - time_since_last_call)
        self._last_api_call = current_time

        model = self._get_gemini_model()
        retries = 4
        backoff = 5
        for attempt in range(1, retries + 1):
            try:
                # Cấu hình an toàn để tránh việc câu trả lời bị cắt do bộ lọc của Google
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]

                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        top_p=0.8,
                        top_k=25,
                    ),
                    safety_settings=safety_settings,
                )
                if hasattr(response, "text") and response.text:
                    return response.text.strip()
                if hasattr(response, "candidates") and response.candidates:
                    parts = response.candidates[0].content.parts
                    full_text = ""
                    for part in parts:
                        if hasattr(part, "text"):
                            full_text += part.text + "\n"
                    if full_text.strip():
                        return full_text.strip()
                return "⚠️ Không nhận được phản hồi hợp lệ từ Gemini."
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    time.sleep(backoff * attempt)
                    continue
                return f"❌ Lỗi khi gọi Gemini: {err}"
        return "Hệ thống đang xử lý nhiều yêu cầu, bạn vui lòng thử lại sau vài phút."

    # ==========================================================================
    # HyDE Query Expansion
    # ==========================================================================

    def hyde_expand_query(self, original_query: str) -> str:
        """Expand query using Hypothetical Document Embeddings."""
        if not self._gemini_ready:
            return original_query
        try:
            hyde_prompt = f"""Bạn là một chuyên gia về quy chế đào tạo đại học.
Hãy viết một đoạn văn ngắn (2-3 câu) trả lời câu hỏi sau như thể bạn đang trích dẫn từ quy chế:

Câu hỏi: {original_query}

Chỉ viết nội dung trả lời, không giải thích."""
            model = self._get_gemini_model()
            response = model.generate_content(
                hyde_prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=200, temperature=0.7),
            )
            if hasattr(response, "text") and response.text:
                hypothetical_doc = response.text.strip()
                return f"{original_query} {hypothetical_doc}"
            return original_query
        except Exception as e:
            print(f"⚠️ HyDE expansion failed: {e}")
            return original_query

    # ==========================================================================
    # Answer Caching
    # ==========================================================================

    def _normalize_cache_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _make_answer_cache_key(self, user_msg: str, recent_history: str) -> str:
        normalized_msg = self._normalize_cache_text(user_msg)
        if not self.config.answer_cache_include_history:
            return f"{self._answer_policy_version}||{normalized_msg}"
        normalized_history = self._normalize_cache_text(recent_history)
        return f"{self._answer_policy_version}||{normalized_msg}||{normalized_history}"

    def _answer_cache_get(self, key: str) -> str | None:
        if self.config.answer_cache_maxsize <= 0:
            return None
        now = time.time()
        with self._cache_lock:
            item = self._answer_cache.get(key)
            if item is None:
                self._answer_cache_misses += 1
                return None
            created_at, answer = item
            if self.config.answer_cache_ttl_seconds > 0 and (now - created_at) > self.config.answer_cache_ttl_seconds:
                self._answer_cache.pop(key, None)
                self._answer_cache_misses += 1
                return None
            self._answer_cache.move_to_end(key)
            self._answer_cache_hits += 1
            return answer

    def _answer_cache_set(self, key: str, answer: str) -> None:
        if self.config.answer_cache_maxsize <= 0:
            return
        now = time.time()
        with self._cache_lock:
            self._answer_cache[key] = (now, answer)
            self._answer_cache.move_to_end(key)
            while len(self._answer_cache) > self.config.answer_cache_maxsize:
                self._answer_cache.popitem(last=False)

    def clear_answer_cache(self) -> None:
        """Clear the answer cache."""
        with self._cache_lock:
            self._answer_cache.clear()
            self._answer_cache_hits = 0
            self._answer_cache_misses = 0

    def get_answer_cache_stats(self) -> dict:
        """Get answer cache statistics."""
        with self._cache_lock:
            size = len(self._answer_cache)
            hits = self._answer_cache_hits
            misses = self._answer_cache_misses
        return {
            "enabled": self.config.answer_cache_maxsize > 0,
            "maxsize": self.config.answer_cache_maxsize,
            "ttl_seconds": self.config.answer_cache_ttl_seconds,
            "include_history": self.config.answer_cache_include_history,
            "size": size,
            "hits": hits,
            "misses": misses,
        }

    def _should_cache_answer(self, answer: str) -> bool:
        a = (answer or "").strip().lower()
        if not a:
            return False
        if a.startswith(("❌", "⚠️")):
            return False
        if "api chưa" in a or "chưa được kết nối" in a or "chưa sẵn sàng" in a:
            return False
        return True

    # ==========================================================================
    # Answer Generation
    # ==========================================================================

    def _generate_answer_uncached(self, user_msg: str, recent_history: str = "") -> str:
        """Generate answer without caching."""
        if not self._gemini_ready:
            return "❌ API chưa được kết nối. Nhấn 'Kết nối API' trong sidebar."
        if self._index is None or not self._records:
            return "❌ Hệ thống chưa sẵn sàng. Vui lòng rebuild database."

        # Query expansion
        word_count = len(user_msg.split())
        vague_keywords = ["sao", "thế nào", "gì", "như nào", "bao nhiêu", "mấy", "được không", "có thể"]
        is_vague = any(kw in user_msg.lower() for kw in vague_keywords)
        expanded_query = self.hyde_expand_query(user_msg) if (word_count < 5 or (word_count < 8 and is_vague)) else user_msg
        query_meta = self._parse_query_metadata(user_msg)
        query_family = self._infer_query_doc_family(user_msg)
        query_families = self._infer_query_doc_families(user_msg)
        is_multi_family_query = len(query_families) >= 2

        # Điều trùng số giữa nhiều văn bản: yêu cầu làm rõ để tránh trả sai nguồn.
        if query_meta.get("article") is not None and query_family is None:
            article_id = str(query_meta["article"])
            family_candidates = sorted(self._article_families.get(article_id, set()))
            if len(family_candidates) > 1:
                family_names = [
                    self._family_display_names.get(f, f.title())
                    for f in family_candidates[:4]
                ]
                family_list = "\n".join([f"- {name}" for name in family_names])
                return (
                    f"📝 Điều {article_id} xuất hiện ở nhiều văn bản khác nhau.\n"
                    "Bạn muốn tra theo văn bản nào sau đây?\n"
                    f"{family_list}\n"
                    "Bạn có thể ghi rõ tên văn bản (ví dụ: Quy chế Học vụ hoặc Quy chế Công tác sinh viên)."
                )

        # Retrieval (15 candidates + two-stage reranking = nhanh hơn 2x)
        retrieve_candidates = self.config.retrieve_candidates   # Default: 15
        rerank_top_k = max(5, self.config.rerank_top_k)         # Ensure we rerank enough
        if query_meta.get("article") is not None or query_family is not None or is_multi_family_query:
            retrieve_candidates = max(retrieve_candidates, 30)
            rerank_top_k = max(rerank_top_k, 8)
        if is_multi_family_query:
            retrieve_candidates = max(retrieve_candidates, 36)
            rerank_top_k = max(rerank_top_k, 10)

        context_records = self.retrieve(expanded_query, top_k=retrieve_candidates)
        if not context_records:
            return "🤔 Tôi chưa tìm thấy thông tin liên quan trong quy chế."

        # Rerank and boost
        reranked_records = self.rerank_results(user_msg, context_records, top_k=rerank_top_k)
        reranked_records = self.apply_metadata_boost(user_msg, reranked_records)
        reranked_records = self._expand_with_cross_document(user_msg, reranked_records, max_extra=8)
        reranked_records = sorted(
            reranked_records,
            key=lambda x: x.get("final_score", x.get("rerank_score", 0.0)),
            reverse=True,
        )
        reranked_records = self._inject_required_context(user_msg, reranked_records, max_extra=4)

        # Nếu user hỏi đích danh "Điều X" nhưng context không chứa Điều X → từ chối
        if query_meta["article"] is not None:
            requested_article = str(query_meta["article"])
            found_in_context = False
            search_window = reranked_records[: min(15, len(reranked_records))]
            for r in search_window:
                # Kiểm tra trong text (header thật)
                if re.search(rf'(?:^|\n)\s*Điều\s+{requested_article}[\s:.]', r.get('text', ''), re.IGNORECASE):
                    found_in_context = True
                    break
                # Kiểm tra metadata
                if str(r.get('article', '')) == requested_article:
                    found_in_context = True
                    break
            if not found_in_context:
                fallback_matches: list[dict] = []
                seen_keys: set[str] = set()
                pat = re.compile(rf'(?:^|\n)\s*Điều\s+{requested_article}[\s:.]', re.IGNORECASE)
                for rec in self._records:
                    if not isinstance(rec, dict):
                        continue
                    t = rec.get("text", "")
                    if pat.search(t) or str(rec.get("article", "")) == requested_article:
                        key = self._make_record_key(rec)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        fallback_matches.append(
                            {
                                **rec,
                                "retrieval_source": "global_exact",
                                "rerank_score": -0.01,
                                "final_score": -0.01,
                            }
                        )
                        if len(fallback_matches) >= 5:
                            break
                if fallback_matches:
                    reranked_records = fallback_matches + reranked_records
                else:
                    return (
                        f"📝 Không tìm thấy thông tin về **Điều {requested_article}** "
                        f"trong dữ liệu quy chế hiện có. "
                        f"Vui lòng kiểm tra lại số Điều hoặc đặt câu hỏi theo nội dung cụ thể."
                    )

        _, confidence_label = self.calculate_confidence(reranked_records)

        # ===== GĐ3A: ContextValidator — kiểm tra đủ bằng chứng TRƯỚC khi gọi LLM =====
        # Chạy SAU bước fallback Điều để tránh báo thiếu dữ liệu giả.
        sufficiency = ContextValidator.check_context_sufficiency(
            user_msg, reranked_records[:5]
        )
        if not sufficiency.get("is_sufficient") and sufficiency.get("recommendation") == "ask_clarification":
            missing = "; ".join(sufficiency.get("missing_info", [])[:2]).strip()
            if missing:
                return (
                    "📝 Ngữ cảnh hiện chưa đủ để trả lời chắc chắn. "
                    f"Thiếu thông tin: {missing}. "
                    "Bạn vui lòng nêu rõ hơn Điều/Mục hoặc nội dung cụ thể cần tra cứu."
                )
            return (
                "📝 Ngữ cảnh hiện chưa đủ để trả lời chắc chắn. "
                "Bạn vui lòng nêu rõ hơn Điều/Mục hoặc nội dung cụ thể cần tra cứu."
            )

        # Take more records for low-confidence or family-sensitive queries.
        top_n = 5
        if confidence_label in ["low", "very_low"] or query_family is not None or is_multi_family_query:
            top_n = 8
        if is_multi_family_query:
            top_n = max(top_n, 10)
        top_records = reranked_records[:top_n]

        # ===== TEXT-FIRST: trích Điều/Mục/tiêu đề từ TEXT thay vì ép metadata =====
        valid_articles: set[str] = set()
        valid_sections: set[str] = set()
        valid_titles: set[str] = set()
        valid_chapters: set[str] = set()
        valid_clauses: set[str] = set()
        valid_families: set[str] = set()
        for r in top_records:
            text_content = r.get("text", "")
            rec_family = str(r.get("doc_family") or "").strip()
            if rec_family:
                valid_families.add(rec_family.lower())

            text_art = re.search(r'(?:^|\n)\s*Điều\s+(\d+)[\s:.]', text_content, re.IGNORECASE)
            if text_art:
                valid_articles.add(text_art.group(1))
            elif r.get("article") is not None:
                valid_articles.add(str(r["article"]))

            text_chap = re.search(r'(?:^|\n)\s*Chương\s+([IVXLCDM]+|\d+)', text_content, re.IGNORECASE)
            if text_chap:
                valid_chapters.add(text_chap.group(1).upper())
            elif r.get("chapter"):
                valid_chapters.add(str(r["chapter"]).upper())

            text_sec = re.search(r'(?:^|\n)\s*Mục\s+([IVXLCDM]+|\d+)', text_content, re.IGNORECASE)
            if text_sec:
                valid_sections.add(text_sec.group(1).upper())
            elif r.get("section"):
                valid_sections.add(str(r["section"]).upper())
            else:
                roman_sec = re.search(r'(?:^|\n)\s*([IVXLCDM]+)\.\s+[^\n]+', text_content, re.IGNORECASE)
                if roman_sec:
                    valid_sections.add(roman_sec.group(1).upper())

            text_clause = re.search(r'(?:^|\n)\s*[Kk]hoản\s+(\d+)\b', text_content)
            if text_clause:
                valid_clauses.add(text_clause.group(1))
            elif r.get("clause") is not None:
                valid_clauses.add(str(r["clause"]))
            else:
                numbered_clause = re.search(r'(?:^|\n)\s*(\d+)\.\s', text_content)
                if numbered_clause:
                    valid_clauses.add(numbered_clause.group(1))

            h_span = str(r.get("header_span") or "").strip()
            if h_span:
                valid_titles.add(h_span.lower())

        has_valid_refs = bool(
            valid_articles
            or valid_sections
            or valid_chapters
            or valid_clauses
            or valid_titles
            or valid_families
        )

        # Mapping nội dung: Chương/Điều/Mục/Phần → snippet để LLM cross-check
        article_content_map: dict[str, str] = {}
        for r in top_records:
            text_content = r.get("text", "")
            text_art = re.search(r'(?:^|\n)\s*Điều\s+(\d+)[\s:.]', text_content, re.IGNORECASE)
            art_key = text_art.group(1) if text_art else (str(r.get("article", "")) if r.get("article") is not None else "")
            text_chap = re.search(r'(?:^|\n)\s*Chương\s+([IVXLCDM]+|\d+)', text_content, re.IGNORECASE)
            chap_key = text_chap.group(1).upper() if text_chap else (str(r.get("chapter", "")) or "")

            text_sec = re.search(r'(?:^|\n)\s*Mục\s+([IVXLCDM]+|\d+)', text_content, re.IGNORECASE)
            if text_sec:
                sec_key = text_sec.group(1).upper()
            elif r.get("section"):
                sec_key = str(r.get("section")).upper()
            else:
                roman_sec = re.search(r'(?:^|\n)\s*([IVXLCDM]+)\.\s+[^\n]+', text_content, re.IGNORECASE)
                sec_key = roman_sec.group(1).upper() if roman_sec else ""

            clause_match = re.search(r'(?:^|\n)\s*[Kk]hoản\s+(\d+)\b', text_content)
            if clause_match:
                clause_key = clause_match.group(1)
            elif r.get("clause") is not None:
                clause_key = str(r.get("clause"))
            else:
                numbered_clause = re.search(r'(?:^|\n)\s*(\d+)\.\s', text_content)
                clause_key = numbered_clause.group(1) if numbered_clause else ""

            h_span = str(r.get("header_span") or "").strip()
            rec_family = str(r.get("doc_family") or "").strip()
            if art_key:
                label = f"Điều {art_key}"
                if chap_key:
                    label = f"Chương {chap_key}, {label}"
                if clause_key:
                    label = f"{label}, Khoản {clause_key}"
            elif sec_key:
                label = f"Mục {sec_key}"
                if clause_key:
                    label = f"{label}, Khoản {clause_key}"
            elif h_span:
                label = f'Phần "{h_span}"'
            elif chap_key:
                label = f"Chương {chap_key}"
            else:
                label = ""

            if label:
                if rec_family:
                    label = f"[{rec_family}] {label}"
                if h_span and not label.endswith(f'"{h_span}"') and label != f'Phần "{h_span}"':
                    label += f' — "{h_span}"'
                if label not in article_content_map:
                    article_content_map[label] = text_content[:300].replace('\n', ' ')

        whitelist_instruction = ""
        if has_valid_refs:
            mapping_lines = "\n".join([f"   - {label}: \"{content}...\"" for label, content in article_content_map.items()][:6])
            whitelist_instruction = (
                f"\n⚠️ CHẾ ĐỘ KIỂM CHỨNG TRÍCH DẪN (STRICT VERIFICATION):\n"
                f"- Tài liệu 'quyche.pdf' bao gồm nhiều văn bản khác nhau. PHẢI xác định đúng tên văn bản trước khi trích dẫn.\n"
                f"- QUY TẮC TRÍCH DẪN:\n"
                f"  1. PHẢI ghi rõ tên văn bản nguồn trước số hiệu.\n"
                f"  2. TRÍCH DẪN THEO ĐỊNH DẠNG GỐC của đoạn truy xuất:\n"
                f"     + Có Chương/Điều thì trích Chương/Điều.\n"
                f"     + Có Mục/Khoản thì trích Mục/Khoản.\n"
                f"     + Không có số hiệu thì trích theo 'Phần [Tiêu đề]'.\n"
                f"  3. TUYỆT ĐỐI KHÔNG tự gán số Điều/Chương/Mục không tồn tại trong đoạn gốc.\n"
                f"{mapping_lines}\n"
            )
        else:
            whitelist_instruction = (
                "\n⚠️ CẢNH BÁO TRÍCH DẪN:\n"
                "- Trong ngữ cảnh chưa có số hiệu rõ ràng.\n"
                "- Không được tự ý thêm/bịa số Điều/Chương/Mục."
            )

        constraint_records = reranked_records[: min(14, len(reranked_records))]
        constraint_report = self._build_constraint_report(user_msg, constraint_records)
        applicable_rules = constraint_report.get("applicable_rules", [])
        exclusion_rules = constraint_report.get("exclusion_rules", [])
        critical_missing_rules = constraint_report.get("critical_missing", [])
        missing_inputs = constraint_report.get("missing_inputs", [])
        decision_query = bool(constraint_report.get("decision_query"))

        # Với câu hỏi dạng ra quyết định cá nhân mà thiếu đầu vào, yêu cầu user bổ sung.
        if decision_query and missing_inputs:
            missing_lines = "\n".join([f"- {m}" for m in missing_inputs[:4]])
            blocking_lines = ""
            if critical_missing_rules:
                blocking_lines = "\nCác điều kiện quan trọng chưa xác minh được từ ngữ cảnh:\n" + "\n".join(
                    [f"- {x}" for x in critical_missing_rules[:3]]
                )
            return (
                "📝 Để kết luận chính xác cho trường hợp của bạn, mình cần thêm dữ liệu sau:\n"
                f"{missing_lines}"
                f"{blocking_lines}\n"
                "Sau khi bạn bổ sung, mình sẽ đối chiếu đầy đủ điều kiện áp dụng và điều kiện loại trừ rồi mới kết luận."
            )

        exclusion_instruction = ""
        if applicable_rules or exclusion_rules or critical_missing_rules:
            apply_lines = "\n".join([f"- {x}" for x in applicable_rules[:6]]) if applicable_rules else "- Chưa trích xuất được điều kiện áp dụng rõ ràng."
            exclude_lines = "\n".join([f"- {x}" for x in exclusion_rules[:6]]) if exclusion_rules else "- Chưa phát hiện điều kiện loại trừ rõ ràng."
            missing_lines = (
                "\nĐiều kiện chưa xác minh được (không được bỏ qua khi kết luận):\n"
                + "\n".join([f"- {x}" for x in critical_missing_rules[:4]])
            ) if critical_missing_rules else ""
            exclusion_instruction = (
                "\n⚙️ BÁO CÁO KIỂM TRA RÀNG BUỘC (CONSTRAINT CHECK):\n"
                "Điều kiện áp dụng:\n"
                f"{apply_lines}\n"
                "Điều kiện loại trừ/ngoại lệ:\n"
                f"{exclude_lines}"
                f"{missing_lines}\n"
                "Trước khi đưa ra kết luận cuối, PHẢI đối chiếu đầy đủ các nhóm điều kiện trên."
            )

        def _format_citation(r: dict) -> str:
            """Text-first citation with document-aware and structure-aware formatting."""
            text_content = r.get("text", "")
            doc_name = r.get("document_name", "")
            doc_family = str(r.get("doc_family") or "").strip()
            source_title = str(r.get("source_title") or "").strip()
            structure_type = str(r.get("structure_type") or "").lower()
            parts: list[str] = []
            source_primary = doc_family or source_title or doc_name or "Văn bản liên quan"

            text_chap = re.search(r'(?:^|\n)\s*Chương\s+([IVXLCDM]+|\d+)', text_content, re.IGNORECASE)
            chap_val = None
            if text_chap:
                chap_val = text_chap.group(1).upper()
            elif r.get("chapter"):
                chap_val = str(r["chapter"]).upper()

            text_sec = re.search(r'(?:^|\n)\s*Mục\s+([IVXLCDM]+|\d+)', text_content, re.IGNORECASE)
            sec_val = None
            if text_sec:
                sec_val = text_sec.group(1).upper()
            elif r.get("section"):
                sec_val = str(r["section"]).upper()
            else:
                roman_sec = re.search(r'(?:^|\n)\s*([IVXLCDM]+)\.\s+[^\n]+', text_content, re.IGNORECASE)
                if roman_sec:
                    sec_val = roman_sec.group(1).upper()

            text_art = re.search(r'(?:^|\n)\s*Điều\s+(\d+)[\s:.]', text_content, re.IGNORECASE)
            art_num = None
            if text_art:
                art_num = text_art.group(1)
            elif structure_type == "article" and r.get("article") is not None:
                art_num = str(r["article"])

            text_clause = re.search(r'(?:^|\n)\s*[Kk]hoản\s+(\d+)\b', text_content)
            clause_num = None
            if text_clause:
                clause_num = text_clause.group(1)
            elif r.get("clause") is not None:
                clause_num = str(r.get("clause"))
            else:
                numbered_clause = re.search(r'(?:^|\n)\s*(\d+)\.\s', text_content)
                if numbered_clause:
                    clause_num = numbered_clause.group(1)

            h_span = str(r.get("header_span") or "").strip()
            if art_num:
                if chap_val:
                    parts.append(f"Chương {chap_val}")
                parts.append(f"Điều {art_num}")
                if clause_num:
                    parts.append(f"Khoản {clause_num}")
            elif sec_val:
                parts.append(f"Mục {sec_val}")
                if clause_num:
                    parts.append(f"Khoản {clause_num}")
            elif chap_val and structure_type == "chapter":
                parts.append(f"Chương {chap_val}")
            elif h_span:
                parts.append(f'Phần "{h_span}"')

            page_start = r.get("page_start")
            page_end = r.get("page_end")
            if page_start is not None:
                if page_end is not None and page_end != page_start:
                    parts.append(f"Trang {page_start}-{page_end}")
                else:
                    parts.append(f"Trang {page_start}")

            source_label = f"[Nguồn: {source_primary}]"
            citation_text = " | ".join([p for p in parts if p])
            return f"{source_label} {citation_text}".strip()

        # GĐ3B: Block boundary rõ ràng cho từng chunk context
        context_blocks = []
        for r in top_records:
            citation = _format_citation(r)
            doc_family = str(r.get("doc_family") or "").strip()
            art_match = re.search(r'(?:^|\n)\s*Điều\s+(\d+)[\s:.]', r.get('text', ''), re.IGNORECASE)
            sec_match = re.search(r'(?:^|\n)\s*Mục\s+([IVXLCDM]+|\d+)', r.get('text', ''), re.IGNORECASE)
            h_span = str(r.get('header_span') or '').strip()
            if art_match:
                unit_label = f'ĐIỀU {art_match.group(1)}'
            elif sec_match:
                unit_label = f'MỤC {sec_match.group(1).upper()}'
            elif r.get("section"):
                unit_label = f"MỤC {str(r.get('section')).upper()}"
            elif h_span:
                unit_label = "PHẦN"
            else:
                unit_label = "KHÁC"
            chap_match = re.search(r'(?:^|\n)\s*Chương\s+([IVXLCDM]+|\d+)', r.get('text', ''), re.IGNORECASE)
            chap_label = f' (Chương {chap_match.group(1).upper()})' if (chap_match and unit_label.startswith("ĐIỀU")) else ''
            family_info = f" [{doc_family}]" if doc_family else ""
            title_info = f' — {h_span}' if h_span else ''
            block = (
                f"\n═════ BẮT ĐẦU NỘI DUNG {unit_label}{chap_label}{family_info}{title_info} ═════\n"
                f"[Score: {r.get('final_score', r.get('rerank_score', 0)):.2f}]\n"
                f"{citation}\n"
                f"Nội dung: {r['text']}\n"
                f"═════ HẾT {unit_label} ═════"
            )
            context_blocks.append(block)
        context_text = "\n".join(context_blocks)

        if confidence_label in ["high", "medium"]:
            confidence_instruction = "\nMỨC TIN CẬY NGỮ CẢNH: CAO. Ưu tiên sử dụng thông tin từ ngữ cảnh truy xuất."
        else:
            confidence_instruction = "\nMỨC TIN CẬY NGỮ CẢNH: THẤP. Trả lời thận trọng, nêu rõ phần nào chưa chắc chắn."
        if not sufficiency.get("is_sufficient") and sufficiency.get("recommendation") == "answer_with_disclaimer":
            missing = "; ".join(sufficiency.get("missing_info", [])[:2]).strip()
            if missing:
                confidence_instruction += f"\nDữ liệu còn thiếu: {missing}. Phải nêu rõ giới hạn thông tin."

        prompt = f"""
Bạn là **trợ lý học vụ HUSC**. Trả lời các câu hỏi về quy chế đào tạo, học phần, tín chỉ, điểm, học lại, bảo lưu, học bổng, tốt nghiệp.

--- NGỮ CẢNH ---
{context_text}

--- LỊCH SỬ ---
{recent_history}

--- CÂU HỎI ---
{user_msg}

--- NGUYÊN TẮC ---
1. Ưu tiên sử dụng ngữ cảnh truy xuất (VectorDB/CSV) nếu liên quan và đủ thông tin.
2. Nếu ngữ cảnh không liên quan hoặc không chứa thông tin cần thiết → trả lời bằng kiến thức học vụ phổ biến và nêu rõ giả định.
3. Không suy diễn ngoài dữ liệu; nếu thiếu thông tin để tính toán (ví dụ GPA thiếu thang điểm, số tín chỉ) → hỏi lại trước khi trả lời.
4. Giữ nguyên URL nếu xuất hiện trong dữ liệu.
5. QUY TẮC BỐ CỤC (ƯU TIÊN DỄ ĐỌC):
   - Câu hỏi ngắn/định nghĩa: trả lời trực tiếp 1-2 đoạn ngắn, không chia mục rườm rà.
   - Câu hỏi phân tích/so sánh/tình huống nhiều điều kiện: ưu tiên 4 mục theo thứ tự:
     ### Kết luận ngắn
     ### Căn cứ chính
     ### Phân tích
     ### Lưu ý (nếu có)
6. QUY TẮC ĐỊNH DẠNG (BẮT BUỘC):
   - Chỉ in đậm cụm nhãn ngắn hoặc thuật ngữ trọng tâm (ví dụ: Kết luận, Căn cứ, Lưu ý, Điều/Khoản).
   - KHÔNG in đậm cả câu dài.
   - Danh sách chỉ dùng 1 cấp bullet, mỗi bullet tối đa 1 ý ngắn.
   - Mỗi đoạn tối đa 2 câu; ưu tiên câu ngắn, mạch lạc.
7. Dấu câu và khoảng trắng phải chuẩn tiếng Việt: một khoảng trắng sau dấu câu, không lặp dấu thừa.
8. Công thức toán học viết bằng LaTeX:
   - Inline: $công thức$
   - Block: $$công thức$$
   - Kèm giải thích các biến nếu cần thiết.
9. Nếu có lời khuyên hoặc lưu ý → đặt ở cuối câu trả lời, tách riêng.
10. Nếu người dùng hỏi về thang điểm/GPA → cung cấp bảng đối chiếu hoặc mô tả thang điểm bằng bảng Markdown khi đủ dữ liệu.
11. CẤU TRÚC TRÍCH DẪN LINH HOẠT: Phải tuân thủ cấu trúc phân cấp của từng văn bản trong nguồn tin.
• Đối với Quy chế học vụ/Công tác sinh viên: trích theo Chương, Điều, Khoản.
• Đối với Quy định học bổng: trích theo Mục (La Mã), Khoản.
• Đối với các chính sách khác: Trích theo Tiêu đề phần.
12. ĐỊNH DANH VĂN BẢN (BẮT BUỘC): Luôn ghi rõ tên văn bản nguồn trước khi trích dẫn số hiệu để tránh nhầm lẫn giữa các quy định trùng số điều.
   Ví dụ: "Theo Quy chế học vụ, Điều 20..." khác với "Theo Quy chế công tác sinh viên, Điều 20...".
13. PHÒNG CHỐNG ẢO GIÁC TUYỆT ĐỐI:
• Nếu không thấy số hiệu Chương/Điều trong đoạn văn bản đang đọc, TUYỆT ĐỐI KHÔNG được tự ý gán số hiệu từ các phần khác vào.
• Trong trường hợp này, hãy dùng cụm từ "Theo quy định tại phần [Tên tiêu đề]...".
• Thà trích dẫn thiếu số hiệu còn hơn trích dẫn sai số hiệu.
14. TRUY SOÁT ĐIỀU KIỆN LOẠI TRỪ (BẮT BUỘC):
• Trước khi chốt kết luận, phải kiểm tra các ngưỡng kỹ thuật, điều kiện loại trừ và ngoại lệ.
• Nếu có điều kiện loại trừ liên quan thì phải nêu tường minh trong mục riêng, không được bỏ qua.
{whitelist_instruction}
{exclusion_instruction}
{confidence_instruction}

--- GỢI Ý CẤU TRÚC TRÌNH BÀY ---
- Khi câu hỏi phức tạp, có nhiều điều kiện: dùng các tiêu đề sau:
  ### Kết luận ngắn
  ### Căn cứ chính
  ### Phân tích
  ### Lưu ý (nếu có)
- Khi câu hỏi đơn giản: trả lời trực tiếp, không cần chia mục.
- Mỗi căn cứ trích dẫn nên bắt đầu bằng "Theo <tên văn bản>, <định dạng gốc> ...".
- Viết câu ngắn, chấm câu chuẩn, không dùng ký hiệu thừa.
"""
        raw_answer = self._ask_gemini(prompt)
        
        # ===== GĐ3C: POST-PROCESSING VALIDATION =====
        # Kiểm tra trích dẫn Điều/Mục có nằm trong whitelist context không
        raw_answer = self._validate_citations(
            raw_answer,
            valid_articles,
            valid_sections,
            valid_chapters,
            valid_clauses,
            valid_titles,
            valid_families,
        )
        raw_answer = self._remove_false_missing_context_claim(raw_answer, user_msg, top_records)
        raw_answer = self._postprocess_answer_markdown(raw_answer)
        
        return raw_answer

    def _validate_citations(
        self,
        answer: str,
        valid_articles: set,
        valid_sections: Optional[set] = None,
        valid_chapters: Optional[set] = None,
        valid_clauses: Optional[set] = None,
        valid_titles: Optional[set] = None,
        valid_families: Optional[set] = None,
    ) -> str:
        """
        GĐ3C: Post-processing — kiểm tra trích dẫn trong câu trả lời.
        """
        valid_sections = set(valid_sections or set())
        valid_chapters = set(valid_chapters or set())
        valid_clauses = set(valid_clauses or set())
        valid_titles = set(valid_titles or set())
        valid_families = set(valid_families or set())
        normalize_title = lambda x: re.sub(r"\s+", " ", str(x).strip().lower())
        valid_titles_norm = {normalize_title(t) for t in valid_titles if str(t).strip()}

        if not valid_articles and not valid_sections and not valid_chapters and not valid_clauses and not valid_titles and not valid_families:
            answer = re.sub(
                r'\*{0,2}Điều\s+\d+\*{0,2}',
                '*(theo quy định)*',
                answer,
                flags=re.IGNORECASE,
            )
            answer = re.sub(
                r'\*{0,2}Mục\s+([IVXLCDM]+|\d+)\*{0,2}',
                '',
                answer,
                flags=re.IGNORECASE,
            )
            answer = re.sub(
                r'\*{0,2}Chương\s+([IVXLCDM]+|\d+)\*{0,2}',
                '',
                answer,
                flags=re.IGNORECASE,
            )
            answer = re.sub(
                r'\*{0,2}Khoản\s+\d+\*{0,2}',
                '',
                answer,
                flags=re.IGNORECASE,
            )
            return answer

        # -----------------------------
        # Validate Điều
        # -----------------------------
        cited_articles = re.findall(r'Điều\s+(\d+)', answer, re.IGNORECASE)
        unique_cited_articles = set(cited_articles)
        if unique_cited_articles:
            if valid_articles:
                invalid_articles = unique_cited_articles - valid_articles
            else:
                invalid_articles = unique_cited_articles

            for art_num in invalid_articles:
                answer = re.sub(
                    rf'\bĐiều\s+{re.escape(str(art_num))}\b',
                    '*(theo quy định)*',
                    answer,
                    flags=re.IGNORECASE,
                )

        # -----------------------------
        # Validate Chương
        # -----------------------------
        cited_chapters = re.findall(r'Chương\s+([IVXLCDM]+|\d+)', answer, re.IGNORECASE)
        unique_cited_chapters = {str(c).upper() for c in cited_chapters}
        valid_chapters_norm = {str(c).upper() for c in valid_chapters}
        if unique_cited_chapters:
            if valid_chapters_norm:
                invalid_chapters = unique_cited_chapters - valid_chapters_norm
            else:
                invalid_chapters = unique_cited_chapters
            for chap in invalid_chapters:
                answer = re.sub(
                    rf'\bChương\s+{re.escape(chap)}\b',
                    '',
                    answer,
                    flags=re.IGNORECASE,
                )

        # -----------------------------
        # Validate Mục
        # -----------------------------
        cited_sections = re.findall(r'Mục\s+([IVXLCDM]+|\d+)', answer, re.IGNORECASE)
        unique_cited_sections = {s.upper() for s in cited_sections}
        valid_sections_norm = {str(s).upper() for s in valid_sections}
        if unique_cited_sections:
            if valid_sections_norm:
                invalid_sections = unique_cited_sections - valid_sections_norm
            else:
                invalid_sections = unique_cited_sections
            for sec in invalid_sections:
                answer = re.sub(
                    rf'\bMục\s+{re.escape(sec)}\b',
                    '',
                    answer,
                    flags=re.IGNORECASE,
                )

        cited_clauses = re.findall(r'Khoản\s+(\d+)', answer, re.IGNORECASE)
        unique_cited_clauses = set(cited_clauses)
        valid_clauses_norm = {str(c) for c in valid_clauses}
        if unique_cited_clauses:
            if valid_clauses_norm:
                invalid_clauses = unique_cited_clauses - valid_clauses_norm
            else:
                invalid_clauses = unique_cited_clauses
            for clause in invalid_clauses:
                answer = re.sub(
                    rf'\bKhoản\s+{re.escape(str(clause))}\b',
                    '',
                    answer,
                    flags=re.IGNORECASE,
                )

        cited_titles_raw: list[str] = []
        cited_titles_raw.extend(re.findall(r'Phần\s+[“"]([^”"\n]{2,160})[”"]', answer, re.IGNORECASE))
        cited_titles_raw.extend(re.findall(r'Phần\s*\[\s*([^\]\n]{2,160})\s*\]', answer, re.IGNORECASE))
        if cited_titles_raw:
            cited_map = [(raw, normalize_title(raw)) for raw in cited_titles_raw if str(raw).strip()]
            cited_titles_norm = {norm for _, norm in cited_map}
            if valid_titles_norm:
                invalid_titles_norm = cited_titles_norm - valid_titles_norm
            else:
                invalid_titles_norm = cited_titles_norm

            invalid_raw_titles = {raw for raw, norm in cited_map if norm in invalid_titles_norm}
            for raw in invalid_raw_titles:
                answer = re.sub(
                    rf'Phần\s+[“"]\s*{re.escape(raw)}\s*[”"]',
                    'Phần liên quan',
                    answer,
                    flags=re.IGNORECASE,
                )
                answer = re.sub(
                    rf'Phần\s*\[\s*{re.escape(raw)}\s*\]',
                    'Phần liên quan',
                    answer,
                    flags=re.IGNORECASE,
                )

        if valid_families:
            valid_family_norm = {normalize_title(x) for x in valid_families if str(x).strip()}
            family_mentions = re.findall(
                r'Theo\s+((?:Quy chế|Quy định|Chế độ|Sổ tay)[^,:\n]{2,120})',
                answer,
                re.IGNORECASE,
            )
            invalid_mentions = []
            canonical_allow = {
                "quy chế học vụ",
                "quy chế công tác sinh viên",
                "quy định học bổng",
                "chế độ, chính sách sinh viên",
                "chế độ chính sách đối với sinh viên",
                "sổ tay sinh viên",
            }
            for fm in family_mentions:
                fm_norm = normalize_title(fm)
                # Skip generic mentions.
                if any(k in fm_norm for k in ["nhà trường", "văn bản", "phần liên quan", "quy định hiện hành", "quy định liên quan"]):
                    continue
                # Giữ lại các tên văn bản chuẩn, tránh đổi thành "Theo văn bản liên quan".
                if any(name in fm_norm for name in canonical_allow):
                    continue
                # Cho phép near-match với family thật trong context (khác dấu câu/độ dài).
                if any((fm_norm in vf) or (vf in fm_norm) for vf in valid_family_norm):
                    continue
                if fm_norm not in valid_family_norm:
                    invalid_mentions.append(fm)
            for fm in set(invalid_mentions):
                answer = re.sub(
                    rf'Theo\s+{re.escape(fm)}',
                    "Theo quy định hiện hành",
                    answer,
                    flags=re.IGNORECASE,
                )

        return answer

    def _normalize_markdown_tables(self, text: str) -> str:
        lines = text.split("\n")
        out: list[str] = []
        i = 0

        def _parse_cells(line: str) -> list[str]:
            raw = line.strip()
            if raw.startswith("|"):
                raw = raw[1:]
            if raw.endswith("|"):
                raw = raw[:-1]
            cells = [c.strip() for c in raw.split("|")]
            return cells

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Bỏ các dòng gạch ngang rác không thuộc bảng markdown hợp lệ.
            if re.fullmatch(r"-{8,}", stripped):
                i += 1
                continue

            if stripped.startswith("|") and stripped.count("|") >= 2:
                header = _parse_cells(stripped)
                if len(header) >= 2 and any(cell for cell in header):
                    out.append("| " + " | ".join(header) + " |")
                    out.append("| " + " | ".join(["---"] * len(header)) + " |")

                    j = i + 1
                    # Bỏ qua separator cũ nếu có
                    if j < len(lines):
                        sep = lines[j].strip()
                        if ("|" in sep and re.fullmatch(r"[:\-\|\s]+", sep)) or re.fullmatch(r"-{8,}", sep):
                            j += 1

                    row_count = 0
                    while j < len(lines):
                        row_line = lines[j].strip()
                        if not row_line:
                            break
                        if re.fullmatch(r"-{8,}", row_line):
                            j += 1
                            continue
                        if "|" not in row_line:
                            break
                        row_cells = _parse_cells(row_line)
                        if len(row_cells) < len(header):
                            row_cells.extend([""] * (len(header) - len(row_cells)))
                        elif len(row_cells) > len(header):
                            row_cells = row_cells[: len(header) - 1] + [" | ".join(row_cells[len(header) - 1 :])]
                        out.append("| " + " | ".join(row_cells) + " |")
                        row_count += 1
                        j += 1

                    if row_count == 0:
                        fallback = [""] * len(header)
                        fallback[0] = "Ghi chú"
                        fallback[-1] = "Thiếu dữ liệu hàng để trình bày bảng."
                        out.append("| " + " | ".join(fallback) + " |")

                    i = j
                    continue

            out.append(line)
            i += 1

        return "\n".join(out)

    def _postprocess_answer_markdown(self, answer: str) -> str:
        if not answer:
            return answer

        label_regex = (
            r"(Kết luận(?: ngắn| cuối cùng)?|Lý do|Căn cứ(?: chính| trích dẫn)?|"
            r"Phân tích|Lưu ý|Điều kiện áp dụng|Điều kiện loại trừ(?:/ngoại lệ)?)"
        )

        allowed_bold_labels = {
            "kết luận",
            "kết luận ngắn",
            "kết luận cuối cùng",
            "lý do",
            "căn cứ",
            "căn cứ chính",
            "căn cứ trích dẫn",
            "phân tích",
            "lưu ý",
            "điều kiện áp dụng",
            "điều kiện loại trừ",
            "điều kiện loại trừ/ngoại lệ",
            "điều",
            "khoản",
            "mục",
            "chương",
        }

        def _normalize_bold(match: re.Match) -> str:
            content = re.sub(r"\s+", " ", match.group(1).strip())
            if not content:
                return ""
            normalized = content.lower().rstrip(":")
            word_count = len(content.split())
            # Chỉ giữ bold cho nhãn ngắn/thuật ngữ ngắn, bỏ bold câu dài gây rối mắt.
            if normalized in allowed_bold_labels or word_count <= 6:
                return f"**{content}**"
            return content

        def _normalize_label_line(match: re.Match) -> str:
            label = re.sub(r"\s+", " ", match.group(1).strip())
            tail = re.sub(r"\s+", " ", (match.group(2) or "").strip())
            if tail:
                return f"**{label}:** {tail}"
            return f"**{label}:**"

        text = answer.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"([^\n])\s+(#{2,6}\s)", r"\1\n\2", text)
        text = re.sub(r"(?m)^[ \t]*[•▪◦]\s+", "- ", text)
        text = re.sub(r"(?m)^[ \t]*\*\s+", "- ", text)
        text = re.sub(r"(?m)^[ \t]{2,}[-*]\s+", "- ", text)
        text = re.sub(r"(?m)^[ \t]{2,}\d+\.\s+", "- ", text)
        text = re.sub(r"\*?\(theo chương liên quan\)\*?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\*?\(theo khoản liên quan\)\*?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\*?\(theo mục liên quan\)\*?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\*?\(?\s*Theo văn bản liên quan(?:[^)\n]{0,80})\)?\*?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\(\s*,\s*", "(", text)
        text = re.sub(r",\s*,", ", ", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = self._normalize_markdown_tables(text)
        text = re.sub(r"\*\*([^*\n]{1,240})\*\*", _normalize_bold, text)
        text = re.sub(
            rf"(?mi)^\s*(?:\*\*)?\s*{label_regex}(?:\*\*)?\s*:\s*(.*)$",
            lambda m: _normalize_label_line(m),
            text,
        )
        text = re.sub(r"(?m)^(#{2,6})([^\s#])", r"\1 \2", text)
        text = re.sub(r"(?m)([^\n])\n(###\s+)", r"\1\n\n\2", text)
        text = re.sub(r"(?m)^(###\s+.+)$", r"\n\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def generate_answer(self, user_msg: str, recent_history: str = "") -> str:
        """Generate answer with Intent Classification and caching support."""
        
        # =====================================================================
        # STEP 1: Intent Classification (trước khi làm bất cứ điều gì)
        # =====================================================================
        if INTENT_CLASSIFIER_AVAILABLE:
            try:
                intent_result = classify_intent(user_msg)
                
                # Nếu là SMALL_TALK hoặc OUT_OF_SCOPE → trả về trực tiếp
                if intent_result.intent != Intent.ACADEMIC:
                    if intent_result.response:
                        return intent_result.response
            except Exception as e:
                # Nếu có lỗi với intent classifier, tiếp tục với RAG pipeline
                print(f"⚠️ Intent classification error: {e}")
        
        ready_for_cache = self._gemini_ready and (self._embedder is not None) and (self._index is not None) and bool(self._records)
        key = self._make_answer_cache_key(user_msg, recent_history)
        if ready_for_cache:
            cached = self._answer_cache_get(key)
            if cached is not None:
                return cached

        answer = self._generate_answer_uncached(user_msg, recent_history)
        if ready_for_cache and self._should_cache_answer(answer):
            self._answer_cache_set(key, answer)
        return answer

# Default singleton instance
_default_engine: Optional[RAGEngine] = None


def _get_default_engine() -> RAGEngine:
    """Get or create the default RAGEngine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = RAGEngine()
    return _default_engine


# Backward-compatible module-level functions
def initialize_once(load_db: bool = True) -> None:
    """Initialize the default RAG engine."""
    _get_default_engine().initialize(load_db)


def build_database() -> bool:
    """Build database using default engine."""
    return _get_default_engine().build_database()


def load_database() -> tuple[list[dict], Any]:
    """Load database using default engine."""
    return _get_default_engine().load_database()


def retrieve_with_vector_store(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve documents using the configured vector backend."""
    return _get_default_engine().retrieve(query, top_k)


def retrieve_with_faiss(query: str, top_k: int = 5) -> list[dict]:
    """Backward-compatible alias for legacy callers."""
    return retrieve_with_vector_store(query, top_k)


def configure_gemini() -> bool:
    """Configure Gemini API using default engine."""
    return _get_default_engine().configure_gemini()


def disconnect_gemini() -> None:
    """Disconnect Gemini API using default engine."""
    _get_default_engine().disconnect_gemini()


def generate_answer(user_msg: str, recent_history: str = "") -> str:
    """Generate answer using default engine."""
    return _get_default_engine().generate_answer(user_msg, recent_history)


def clear_answer_cache() -> None:
    """Clear answer cache of default engine."""
    _get_default_engine().clear_answer_cache()


def get_answer_cache_stats() -> dict:
    """Get cache stats from default engine."""
    return _get_default_engine().get_answer_cache_stats()


# Legacy variable access (for api_chat.py compatibility)
CSV_FILE = RAGConfig().csv_file
PDF_FILE = RAGConfig().pdf_file


def __getattr__(name: str):
    """Module-level __getattr__ for backward compatibility with dynamic attributes."""
    engine = _get_default_engine()
    if name == "gemini_ready":
        return engine.gemini_ready
    elif name == "embedder":
        return engine.embedder
    elif name == "index":
        return engine.index
    elif name == "records":
        return engine.records
    elif name == "bm25_index":
        return engine.bm25_index
    elif name == "vector_db_type":
        return engine.vector_db_type
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
