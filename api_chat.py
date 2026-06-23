import asyncio
import os
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import rag_core

# ==============================================================================
# RATE LIMITER IMPORT
# ==============================================================================
try:
    from rate_limiter import get_rate_limiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False
    get_rate_limiter = None  # type: ignore


app = FastAPI(title="Chat API")


class ChatRequest(BaseModel):
    message: str
    recent_history: str | None = ""


class ChatResponse(BaseModel):
    reply: str
    latency_ms: int


@app.on_event("startup")
def _startup() -> None:
    rag_core.initialize_once(load_db=True)


@app.get("/status")
def status():
    csv_path = rag_core.CSV_FILE
    pdf_path = rag_core.PDF_FILE
    vector_db_type = str(getattr(rag_core, "vector_db_type", "faiss")).lower()
    vector_store_ready = bool(rag_core.index) and bool(rag_core.records)
    source_counts: dict[str, int] = {}
    try:
        for r in rag_core.records or []:
            if isinstance(r, dict):
                src = str(r.get("source") or "")
                source_counts[src] = source_counts.get(src, 0) + 1
    except Exception:
        source_counts = {}
    return {
        "gemini_ready": rag_core.gemini_ready,
        "vector_db_type": vector_db_type,
        "vector_store_ready": vector_store_ready,
        "faiss_ready": vector_db_type == "faiss" and vector_store_ready,
        "qdrant_ready": vector_db_type == "qdrant" and vector_store_ready,
        "bm25_ready": bool(rag_core.bm25_index) and bool(rag_core.records),
        "retrieval_ready": bool(rag_core.records) and (bool(rag_core.bm25_index) or vector_store_ready),
        "embedder_ready": rag_core.embedder is not None,
        "records_count": len(rag_core.records) if rag_core.records else 0,
        "csv_path": csv_path,
        "csv_exists": os.path.exists(csv_path),
        "pdf_path": pdf_path,
        "pdf_exists": os.path.exists(pdf_path),
        "source_counts": source_counts,
    }


@app.get("/performance_stats")
def performance_stats():
    """Get performance statistics from RAG Engine."""
    return rag_core.get_performance_stats()


@app.post("/performance_reset")
def performance_reset():
    """Reset performance statistics."""
    rag_core.clear_performance_stats()
    return {"success": True}


@app.get("/cache_stats")
def cache_stats():
    return rag_core.get_answer_cache_stats()


@app.post("/cache_clear")
def cache_clear():
    rag_core.clear_answer_cache()
    return {"success": True}


@app.post("/connect_api")
async def connect_api():
    ok = await asyncio.to_thread(rag_core.configure_gemini)
    if not ok:
        raise HTTPException(status_code=400, detail="Không thể kết nối Gemini. Kiểm tra GEMINI_API_KEY.")
    return {"success": True, "message": "✅ Kết nối API thành công!"}


@app.post("/disconnect_api")
def disconnect_api():
    rag_core.disconnect_gemini()
    return {"success": True, "message": "✅ Đã ngắt kết nối API!"}


@app.post("/rebuild_db")
async def rebuild_db():
    ok = await asyncio.to_thread(rag_core.build_database)
    if ok:
        await asyncio.to_thread(rag_core.load_database)
        return {"success": True, "message": "✅ Đã rebuild database thành công!"}
    raise HTTPException(status_code=500, detail="Không thể rebuild database")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Chat endpoint with rate limiting and security logging."""
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-ID")
    
    # Rate limiting check
    if RATE_LIMITER_AVAILABLE:
        limiter = get_rate_limiter()
        rate_result = limiter.check_rate_limit(ip=client_ip, user_id=user_id)
        
        if not rate_result.allowed:
            raise HTTPException(
                status_code=429, 
                detail=rate_result.message,
                headers={"Retry-After": str(rate_result.retry_after or 60)}
            )

    start = time.perf_counter()
    reply = await asyncio.to_thread(rag_core.generate_answer, msg, req.recent_history or "")
    latency_ms = int((time.perf_counter() - start) * 1000)
    
    return ChatResponse(reply=reply, latency_ms=latency_ms)


# ==============================================================================
# SECURITY ENDPOINTS
# ==============================================================================

@app.get("/security/stats")
def security_stats():
    """Get security statistics."""
    result = {}
    
    if RATE_LIMITER_AVAILABLE:
        limiter = get_rate_limiter()
        result["rate_limiter"] = limiter.get_statistics()
    
    return result


@app.get("/security/ip/{ip}")
def ip_status(ip: str):
    """Get status for specific IP."""
    if not RATE_LIMITER_AVAILABLE:
        raise HTTPException(status_code=501, detail="Rate limiter not available")
    
    limiter = get_rate_limiter()
    return limiter.get_ip_status(ip)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST") or "0.0.0.0"
    port = int(os.getenv("PORT") or "8000")
    uvicorn.run(
        "api_chat:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "").lower() in ("1", "true", "yes"),
    )
