# Copilot Instructions for AI Agents

## Project Overview
- **Purpose:** This project is a hybrid chatbot platform for academic Q&A, combining a Flask web UI (user auth, session, chat history, email/OTP flows) with a FastAPI backend for RAG/LLM-based answer generation.
- **Major Components:**
  - `upgraded_text_classifier.py`: Flask app for user management, chat UI, and SQL Server integration.
  - `api_chat.py`: FastAPI app exposing endpoints for chat, status, cache, and database rebuild.
  - `rag_core.py`: Core RAG logic (retrieval, embedding, reranking, Gemini API, PDF/CSV ingestion, answer caching).
  - HTML templates: User-facing pages for login, registration, chat, password reset, etc.

## Architecture & Data Flow
- **User Flow:**
  1. User interacts with Flask UI (`upgraded_text_classifier.py`).
  2. Flask stores chat/user data in SQL Server and calls FastAPI endpoints for chat responses.
  3. FastAPI (`api_chat.py`) uses `rag_core.py` to retrieve/generate answers, leveraging FAISS/BM25, reranking, and Gemini LLM.
- **Data Sources:**
  - QA pairs from CSV, regulatory text from PDF (see `rag_core.py` for file paths).
  - Embeddings and indexes are cached on disk for fast retrieval.

## Developer Workflows
- **Run Flask UI:**
  - `python upgraded_text_classifier.py` (default port 5000)
- **Run FastAPI backend:**
  - `python api_chat.py` (default port 8000, uses uvicorn)
- **Environment:**
  - Set required secrets in `.env` or `.env.txt` (see `rag_core.py` for keys like `GEMINI_API_KEY`, `SECRET_KEY`, etc.)
- **Database:**
  - SQL Server required for user/chat tables (see `get_connection()` in `upgraded_text_classifier.py`).
- **Rebuild RAG DB:**
  - POST `/rebuild_db` to FastAPI or use UI button (triggers CSV/PDF ingestion and index rebuild).

## Project-Specific Patterns
- **User emails must end with `.edu` or `.edu.vn` (see registration logic).
- **Password hashes are upgraded on login if legacy SHA256 detected.
- **CSRF protection for key endpoints (see `CSRF_PROTECTED_ENDPOINTS`).
- **Chat history is stored per user in SQL Server, not in memory.
- **RAG answer cache is LRU with optional TTL, controlled by env vars.
- **Gemini API connection is managed via explicit connect/disconnect endpoints.

## Integration Points
- **External:**
  - Google Gemini API (via `google.generativeai`)
  - SentenceTransformers, FAISS, BM25, PyPDF2, Flask-Mail, pyodbc
- **Internal:**
  - Flask <-> FastAPI via HTTP (see `FASTAPI_BASE_URL`)

## Conventions
- **All major logic is in Python; no JS SPA or frontend build step.**
- **Sensitive config/secrets must be in env files, not hardcoded.**
- **All user-facing text is in Vietnamese.**
- **Error messages and logs use emoji for clarity.**

## Key Files
- `upgraded_text_classifier.py`: Flask app, user/session/chat logic
- `api_chat.py`: FastAPI endpoints
- `rag_core.py`: RAG, embedding, Gemini, cache
- HTML files: UI templates

---
_If any section is unclear or missing, please provide feedback for further refinement._
