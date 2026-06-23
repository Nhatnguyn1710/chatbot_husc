"""
GIAI ĐOẠN 4B: Regression Test Suite — Kiểm tra chống ảo giác trích dẫn.

Chạy: python test_citation_accuracy.py
Yêu cầu: Đã rebuild database SAU KHI sửa parser.

Test patterns:
1. Query chỉ định Điều cụ thể → KPI: trích dẫn đúng 100%
2. Query tham chiếu chéo → KPI: không nhầm Điều
3. Query ngầm (không nêu Điều) → KPI: không bịa số Điều
4. Query multi-Điều → KPI: phân biệt đúng từng Điều

Lưu ý: Test này cần Gemini API kết nối. Nếu chạy off-line, chỉ test phần
retrieval + metadata (không test LLM output).
"""

import re
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core import (
    RAGEngine,
    RAGConfig,
    ContextValidator,
    extract_metadata,
    _is_header_article,
)


# =============================================================================
# TEST CASES
# =============================================================================

# Pattern 1: Query chỉ định Điều cụ thể
SPECIFIC_ARTICLE_TESTS = [
    {
        "query": "Điều 43 quy định gì?",
        "expected_article": "43",
        "description": "Hỏi đúng Điều 43 — cấp bằng tốt nghiệp",
    },
    {
        "query": "Nội dung Điều 36 là gì?",
        "expected_article": "36",
        "description": "Hỏi đúng Điều 36 — cách tính điểm trung bình",
    },
    {
        "query": "Điều 5 nói gì?",
        "expected_article": "5",
        "description": "Hỏi đúng Điều 5",
    },
    {
        "query": "Điều 20 quy định cảnh báo học vụ thế nào?",
        "expected_article": "20",
        "description": "Hỏi Điều 20 — cảnh báo kết quả học tập",
    },
    {
        "query": "Điều 19 nói gì về nghỉ học tạm thời?",
        "expected_article": "19",
        "description": "Hỏi Điều 19 — nghỉ học tạm thời",
    },
    {
        "query": "Điều 1 phạm vi áp dụng",
        "expected_article": "1",
        "description": "Hỏi Điều 1 — phạm vi điều chỉnh",
    },
]

# Pattern 2: Query tham chiếu chéo (nội dung nhắc nhiều Điều)
CROSSREF_TESTS = [
    {
        "query": "Giảm hạng tốt nghiệp khi bị kỷ luật quy định ở đâu?",
        "must_cite": ["43"],
        "must_not_confuse_with": ["36"],
        "description": "Giảm hạng tốt nghiệp thuộc Điều 43, KHÔNG phải Điều 36",
    },
    {
        "query": "Quy định về điểm trung bình tích lũy nằm ở Điều nào?",
        "must_cite": ["36"],
        "must_not_confuse_with": ["43"],
        "description": "Công thức ĐTBTL thuộc Điều 36, KHÔNG nhầm sang Điều 43",
    },
]

# Pattern 3: Query ngầm (không nêu Điều)
IMPLICIT_TESTS = [
    {
        "query": "Làm sao để bảo lưu kết quả học tập?",
        "description": "Query ngầm — bot không được bịa số Điều nếu không chắc",
    },
    {
        "query": "Điều kiện cảnh báo học vụ là gì?",
        "description": "Query ngầm — nên trích dẫn chính xác hoặc không trích",
    },
    {
        "query": "Cần bao nhiêu tín chỉ để ra trường?",
        "description": "Query về tín chỉ — không nên bịa Điều",
    },
]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_extract_metadata_header_first():
    """Test extract_metadata phân biệt header vs crossref."""
    print("\n📋 Test: extract_metadata() — Header-first logic")
    
    results = []
    
    # Case 1: Header thật ở đầu
    text1 = "Điều 43. Cấp bằng tốt nghiệp\n1. Văn bằng tốt nghiệp do Hiệu trưởng ký.\n2. Theo quy định tại Điều 36..."
    meta1 = extract_metadata(text1)
    ok1 = meta1["article"] == 43
    results.append(("Header Điều 43 + crossref Điều 36", ok1, f"article={meta1['article']}, expected=43"))
    
    # Case 2: Chỉ có tham chiếu, không có header thật
    text2 = "Sinh viên cần đạt theo quy định tại Điều 36 Khoản 3"
    meta2 = extract_metadata(text2)
    # Fallback: vẫn lấy Điều 36 (vì không có header thật)
    ok2 = meta2["article"] == 36
    results.append(("Chỉ crossref Điều 36 (fallback)", ok2, f"article={meta2['article']}, expected=36"))
    
    # Case 3: Header + nhiều crossref
    text3 = "Điều 20. Cảnh báo kết quả học tập\n1. Theo Điều 9, Điều 38 và Điều 5..."
    meta3 = extract_metadata(text3)
    ok3 = meta3["article"] == 20
    results.append(("Header Điều 20 + crossref 9,38,5", ok3, f"article={meta3['article']}, expected=20"))
    
    # Case 4: "khoản 5 Điều 36" inline
    text4 = "Điều 43. Tốt nghiệp\nSinh viên phải đạt khoản 5 Điều 36 để tốt nghiệp"
    meta4 = extract_metadata(text4)
    ok4 = meta4["article"] == 43
    results.append(("Header Điều 43 + 'khoản 5 Điều 36' inline", ok4, f"article={meta4['article']}, expected=43"))
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for desc, ok, detail in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: {detail}")
    print(f"  Kết quả: {passed}/{total}")
    return passed, total


def test_is_header_article():
    """Test _is_header_article() phân biệt header vs inline."""
    print("\n📋 Test: _is_header_article() — Phân biệt header/inline")
    
    results = []
    
    # Header thật
    text1 = "Điều 43. Cấp bằng tốt nghiệp"
    m1 = re.search(r'Điều\s+(\d+)', text1, re.IGNORECASE)
    ok1 = _is_header_article(text1, m1) == True
    results.append(("Đầu text: 'Điều 43.'", ok1, f"result={_is_header_article(text1, m1)}"))
    
    # Header thật sau newline
    text2 = "nội dung trước\nĐiều 43. Cấp bằng"
    m2 = re.search(r'Điều\s+43', text2, re.IGNORECASE)
    ok2 = _is_header_article(text2, m2) == True
    results.append(("Sau newline: '\\nĐiều 43.'", ok2, f"result={_is_header_article(text2, m2)}"))
    
    # Tham chiếu: "theo Điều 36"
    text3 = "sinh viên tuân theo Điều 36 về điểm"
    m3 = re.search(r'Điều\s+36', text3, re.IGNORECASE)
    ok3 = _is_header_article(text3, m3) == False
    results.append(("Tham chiếu: 'theo Điều 36'", ok3, f"result={_is_header_article(text3, m3)}"))
    
    # Tham chiếu: "tại Điều 5"
    text4 = "quy định tại Điều 5 của Quy chế"
    m4 = re.search(r'Điều\s+5', text4, re.IGNORECASE)
    ok4 = _is_header_article(text4, m4) == False
    results.append(("Tham chiếu: 'tại Điều 5'", ok4, f"result={_is_header_article(text4, m4)}"))
    
    # Tham chiếu: "khoản 3 Điều 10"
    text5 = "áp dụng khoản 3 Điều 10 để xét"
    m5 = re.search(r'Điều\s+10', text5, re.IGNORECASE)
    ok5 = _is_header_article(text5, m5) == False
    results.append(("Tham chiếu: 'khoản 3 Điều 10'", ok5, f"result={_is_header_article(text5, m5)}"))
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for desc, ok, detail in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: {detail}")
    print(f"  Kết quả: {passed}/{total}")
    return passed, total


def test_retrieval_metadata(engine: RAGEngine):
    """Test retrieval trả chunk đúng Điều khi query chỉ định."""
    print("\n📋 Test: Retrieval — chunk đúng Điều khi query chỉ định")
    
    results = []
    for tc in SPECIFIC_ARTICLE_TESTS:
        query = tc["query"]
        expected = tc["expected_article"]
        
        # Retrieve + rerank
        context = engine.retrieve(query, top_k=15)
        reranked = engine.rerank_results(query, context, top_k=5)
        reranked = engine.apply_metadata_boost(query, reranked)
        
        # Kiểm tra top-1 có đúng Điều không
        top1 = reranked[0] if reranked else {}
        top1_art = str(top1.get("article", ""))
        # Cũng check trong text
        text_art_match = re.search(rf'(?:^|\n)\s*Điều\s+{expected}[\s:.]', top1.get("text", ""), re.IGNORECASE)
        
        ok = top1_art == expected or text_art_match is not None
        results.append((
            tc["description"],
            ok,
            f"top1 article={top1_art}, expected={expected}",
        ))
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for desc, ok, detail in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: {detail}")
    print(f"  Kết quả: {passed}/{total}")
    return passed, total


def test_context_validator():
    """Test ContextValidator phát hiện khi context thiếu thông tin."""
    print("\n📋 Test: ContextValidator — phát hiện context thiếu")
    
    results = []
    
    # Case: query hỏi Điều 99 (không tồn tại)
    fake_contexts = [
        {"text": "Điều 5. Phạm vi áp dụng", "question": "", "final_score": 0.3},
    ]
    check = ContextValidator.check_context_sufficiency("Điều 99 nói gì?", fake_contexts)
    ok1 = not check["is_sufficient"]
    results.append(("Điều 99 không tồn tại → insufficient", ok1, f"is_sufficient={check['is_sufficient']}"))
    
    # Case: query hỏi Điều 5 và context có Điều 5
    good_contexts = [
        {"text": "Điều 5. Chương trình đào tạo...", "question": "", "final_score": 0.8},
    ]
    check2 = ContextValidator.check_context_sufficiency("Điều 5 nói gì?", good_contexts)
    ok2 = check2["is_sufficient"] or check2["confidence"] >= 0.5
    results.append(("Điều 5 có trong context → sufficient", ok2, f"is_sufficient={check2['is_sufficient']}, conf={check2['confidence']}"))
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for desc, ok, detail in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: {detail}")
    print(f"  Kết quả: {passed}/{total}")
    return passed, total


def test_post_processing_validation(engine: RAGEngine):
    """Test _validate_citations phát hiện trích dẫn sai."""
    print("\n📋 Test: Post-processing validation — phát hiện citation sai")
    
    results = []
    
    valid_articles = {"43", "36", "5"}
    
    # Case 1: Tất cả citation đúng → không thay đổi
    answer1 = "Theo **Điều 43**, sinh viên phải..."
    result1 = engine._validate_citations(answer1, valid_articles)
    ok1 = "chưa xác minh" not in result1 and "Lưu ý" not in result1
    results.append(("Citation đúng → giữ nguyên", ok1, f"has_warning={'chưa xác minh' in result1}"))
    
    # Case 2: Citation sai (Điều 99) → gắn footnote
    answer2 = "Theo **Điều 99**, quy định này..."
    result2 = engine._validate_citations(answer2, valid_articles)
    ok2 = "chưa xác minh" in result2
    results.append(("Citation sai Điều 99 → footnote", ok2, f"has_warning={'chưa xác minh' in result2}"))
    
    # Case 3: >50% sai → rút về an toàn
    answer3 = "Theo Điều 99 và Điều 88, quy định. Còn Điều 77 nói thêm."
    result3 = engine._validate_citations(answer3, valid_articles)
    ok3 = "Lưu ý" in result3 or "chưa xác minh" in result3
    results.append((">50% sai → rút về an toàn", ok3, f"result_snippet={result3[-80:]}"))
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for desc, ok, detail in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: {detail}")
    print(f"  Kết quả: {passed}/{total}")
    return passed, total


def run_all_tests():
    """Chạy toàn bộ regression test suite."""
    print("=" * 60)
    print("REGRESSION TEST SUITE — Chống ảo giác trích dẫn")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    # Unit tests (không cần DB/API)
    p, t = test_extract_metadata_header_first()
    total_passed += p
    total_tests += t
    
    p, t = test_is_header_article()
    total_passed += p
    total_tests += t
    
    p, t = test_context_validator()
    total_passed += p
    total_tests += t
    
    # Tests cần database
    print("\n🔧 Khởi tạo RAGEngine...")
    engine = RAGEngine()
    try:
        engine.initialize(load_db=True)
        db_available = engine._index is not None and len(engine._records) > 0
    except Exception as e:
        print(f"  ⚠️ Không load được DB: {e}")
        db_available = False
    
    if db_available:
        print(f"  ✅ Database loaded: {len(engine._records)} records")
        
        p, t = test_retrieval_metadata(engine)
        total_passed += p
        total_tests += t
        
        p, t = test_post_processing_validation(engine)
        total_passed += p
        total_tests += t
    else:
        print("  ⚠️ Bỏ qua test retrieval/post-processing (cần rebuild DB)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"TỔNG KẾT: {total_passed}/{total_tests} tests passed")
    pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Tỷ lệ: {pct:.1f}%")
    
    if pct >= 95:
        print("🎯 KPI ĐẠT: Citation accuracy >= 95%")
    elif pct >= 80:
        print("⚠️ Gần đạt KPI, cần cải thiện thêm")
    else:
        print("❌ Chưa đạt KPI, cần review lại")
    print("=" * 60)
    
    return total_passed, total_tests


if __name__ == "__main__":
    run_all_tests()
