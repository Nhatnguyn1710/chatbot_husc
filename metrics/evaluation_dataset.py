
RETRIEVAL_TESTS = [
    # --- Câu hỏi chỉ định Điều cụ thể (từ PDF quy chế) ---
    {
        "query": "Điều 43 quy định gì?",
        "expected_articles": ["43"],
        "expected_sources": ["pdf"],
        "category": "specific_article",
        "description": "Hỏi đúng Điều 43 — cấp bằng tốt nghiệp",
    },
    {
        "query": "Nội dung Điều 36 là gì?",
        "expected_articles": ["36"],
        "expected_sources": ["pdf"],
        "category": "specific_article",
        "description": "Hỏi đúng Điều 36 — cách tính điểm trung bình",
    },
    {
        "query": "Điều 5 nói gì?",
        "expected_articles": ["5"],
        "expected_sources": ["pdf"],
        "category": "specific_article",
        "description": "Hỏi đúng Điều 5",
    },
    {
        "query": "Điều 20 quy định cảnh báo học vụ thế nào?",
        "expected_articles": ["20"],
        "expected_sources": ["pdf"],
        "category": "specific_article",
        "description": "Hỏi Điều 20 — cảnh báo kết quả học tập",
    },
    {
        "query": "Điều 19 nói gì về nghỉ học tạm thời?",
        "expected_articles": ["19"],
        "expected_sources": ["pdf"],
        "category": "specific_article",
        "description": "Hỏi Điều 19 — nghỉ học tạm thời",
    },
    {
        "query": "Điều 1 phạm vi áp dụng",
        "expected_articles": ["1"],
        "expected_sources": ["pdf"],
        "category": "specific_article",
        "description": "Hỏi Điều 1 — phạm vi điều chỉnh",
    },
    # --- Câu hỏi ngầm (không nêu Điều) ---
    {
        "query": "Điều kiện cảnh báo học vụ là gì?",
        "expected_articles": ["20"],
        "expected_sources": ["pdf"],
        "category": "implicit",
        "description": "Query ngầm về cảnh báo học vụ → nên tìm Điều 20",
    },
    {
        "query": "Cần bao nhiêu tín chỉ để ra trường?",
        "expected_articles": [],
        "expected_sources": ["csv"],
        "category": "implicit",
        "description": "Query từ CSV QA — thông tin tín chỉ tốt nghiệp",
    },
    {
        "query": "Điều kiện nhận học bổng khuyến khích học tập",
        "expected_articles": [],
        "expected_sources": ["csv", "pdf"],
        "category": "implicit",
        "description": "Query về học bổng — có thể từ cả CSV và PDF",
    },
    {
        "query": "Làm sao để bảo lưu kết quả học tập?",
        "expected_articles": ["19"],
        "expected_sources": ["pdf"],
        "category": "implicit",
        "description": "Query ngầm về bảo lưu → liên quan Điều 19",
    },
    # --- Tham chiếu chéo ---
    {
        "query": "Giảm hạng tốt nghiệp khi bị kỷ luật quy định ở đâu?",
        "expected_articles": ["43"],
        "expected_sources": ["pdf"],
        "category": "crossref",
        "description": "Giảm hạng tốt nghiệp thuộc Điều 43",
    },
    {
        "query": "Quy định về điểm trung bình tích lũy nằm ở Điều nào?",
        "expected_articles": ["36"],
        "expected_sources": ["pdf"],
        "category": "crossref",
        "description": "Công thức ĐTBTL thuộc Điều 36",
    },
    # --- Câu hỏi từ CSV (kiến thức CNTT chung) ---
    {
        "query": "Học CNTT cần những yếu tố gì?",
        "expected_articles": [],
        "expected_sources": ["csv"],
        "category": "csv_qa",
        "description": "QA từ CSV — kiến thức CNTT",
    },
    {
        "query": "Nên bắt đầu với ngôn ngữ lập trình nào?",
        "expected_articles": [],
        "expected_sources": ["csv"],
        "category": "csv_qa",
        "description": "QA từ CSV — ngôn ngữ lập trình",
    },
    {
        "query": "Học lại và học cải thiện khác nhau thế nào?",
        "expected_articles": [],
        "expected_sources": ["csv", "pdf"],
        "category": "csv_qa",
        "description": "QA từ CSV — học lại vs cải thiện",
    },
]


# =============================================================================
# 2. GENERATION TEST SET (End-to-end: query → answer)
#    Mỗi entry gồm: query, golden_answer (câu trả lời chuẩn),
#    key_facts (các fact bắt buộc phải có trong answer)
# =============================================================================

GENERATION_TESTS = [
    {
        "query": "Tích lũy bao nhiêu tín chỉ là đủ điều kiện ra trường?",
        "golden_answer": "Riêng với ngành CNTT thì sinh viên phải tích lũy đủ 123 tín chỉ trong đó có 95 tín chỉ bắt buộc và 68 tín chỉ tự chọn.",
        "key_facts": ["123 tín chỉ", "95 tín chỉ bắt buộc"],
        "category": "factual",
    },
    {
        "query": "Chuẩn đầu ra Tiếng Anh của khoa mình là gì?",
        "golden_answer": "Chuẩn đầu ra tiếng anh theo quy định của nhà trường là đạt chứng nhận từ B1 trở lên, với toeic phải đạt 450 điểm trở lên và Ielts là từ 4.5 trở lên.",
        "key_facts": ["B1", "450", "4.5"],
        "category": "factual",
    },
    {
        "query": "Điều kiện để nhận học bổng khuyến khích học tập là gì?",
        "golden_answer": "Học bổng khuyến khích học tập được xét dựa trên kết quả học tập và rèn luyện của học kỳ trước đó. Sinh viên cần có điểm trung bình học kỳ từ loại Khá (GPA >= 2.5) và điểm rèn luyện từ loại Tốt (>= 80 điểm) trở lên.",
        "key_facts": ["GPA", "2.5", "80 điểm", "rèn luyện"],
        "category": "factual",
    },
    {
        "query": "Khi nào bị cảnh báo học vụ?",
        "golden_answer": "Bạn bị cảnh báo học vụ nếu điểm trung bình học kỳ dưới 0,80 ở học kỳ đầu của khóa hoặc dưới 1,00 ở các học kỳ tiếp theo.",
        "key_facts": ["0,80", "1,00", "cảnh báo"],
        "category": "factual",
    },
    {
        "query": "Học lại và học cải thiện khác nhau thế nào?",
        "golden_answer": "Học lại là đăng ký học lại học phần đã rớt điểm F. Học cải thiện là tự nguyện đăng ký học lại học phần đã qua để nâng điểm trung bình. Điểm cao nhất trong các lần học sẽ được dùng để tính điểm trung bình tích lũy.",
        "key_facts": ["học lại", "học cải thiện", "điểm F", "điểm cao nhất"],
        "category": "comparison",
    },
    {
        "query": "Thời gian tối đa hoàn thành chương trình là bao lâu?",
        "golden_answer": "Thời gian tối đa để hoàn thành một chương trình không vượt quá hai lần thời gian theo kế hoạch học tập chuẩn. Chương trình 4 năm thì tối đa 8 năm.",
        "key_facts": ["hai lần", "4 năm", "8 năm"],
        "category": "factual",
    },
    {
        "query": "Nếu bị rớt một môn học bắt buộc thì phải làm sao?",
        "golden_answer": "Nếu bị rớt một môn học bắt buộc, bạn phải đăng ký học lại môn đó ở các học kỳ tiếp theo cho đến khi đạt.",
        "key_facts": ["đăng ký học lại", "học kỳ tiếp theo"],
        "category": "procedural",
    },
    {
        "query": "Nên học ngôn ngữ lập trình nào?",
        "golden_answer": "Dễ nhất là Python. Muốn làm web thì học JavaScript. Thích backend mạnh thì Java hoặc Go. Làm mobile: Kotlin (Android) hoặc Swift (iOS).",
        "key_facts": ["Python", "JavaScript"],
        "category": "advisory",
    },
]


# =============================================================================
# 3. INTENT CLASSIFICATION TEST SET
#    Mỗi entry gồm: query, expected_intent (ACADEMIC/SMALL_TALK/OUT_OF_SCOPE)
# =============================================================================

INTENT_TESTS = [
    # --- SMALL_TALK ---
    {"query": "Xin chào", "expected_intent": "small_talk", "category": "greeting"},
    {"query": "Hello", "expected_intent": "small_talk", "category": "greeting"},
    {"query": "Hi", "expected_intent": "small_talk", "category": "greeting"},
    {"query": "Cảm ơn bạn", "expected_intent": "small_talk", "category": "thanks"},
    {"query": "Tạm biệt", "expected_intent": "small_talk", "category": "goodbye"},
    {"query": "Bạn là ai?", "expected_intent": "small_talk", "category": "bot_identity"},
    {"query": "Bạn khỏe không?", "expected_intent": "small_talk", "category": "how_are_you"},
    {"query": "Ok", "expected_intent": "small_talk", "category": "acknowledgment"},
    {"query": "Oke", "expected_intent": "small_talk", "category": "acknowledgment"},
    # --- OUT_OF_SCOPE ---
    {"query": "Hôm nay thời tiết thế nào?", "expected_intent": "out_of_scope", "category": "weather"},
    {"query": "Phim hay nhất năm nay là gì?", "expected_intent": "out_of_scope", "category": "entertainment"},
    {"query": "Quán ăn ngon ở Huế?", "expected_intent": "out_of_scope", "category": "food"},
    {"query": "Tôi đang buồn quá", "expected_intent": "out_of_scope", "category": "emotional"},
    {"query": "Bóng đá hôm nay ai thắng?", "expected_intent": "out_of_scope", "category": "entertainment"},
    # --- ACADEMIC ---
    {"query": "Điều 43 quy định gì?", "expected_intent": "academic", "category": "article_specific"},
    {"query": "Điều kiện cảnh báo học vụ là gì?", "expected_intent": "academic", "category": "regulation"},
    {"query": "Tín chỉ tối thiểu mỗi kỳ là bao nhiêu?", "expected_intent": "academic", "category": "credit"},
    {"query": "GPA bao nhiêu thì được học bổng?", "expected_intent": "academic", "category": "scholarship"},
    {"query": "Cách tính điểm trung bình tích lũy?", "expected_intent": "academic", "category": "gpa"},
    {"query": "Làm sao để bảo lưu kết quả học tập?", "expected_intent": "academic", "category": "procedure"},
    {"query": "Học cải thiện là gì?", "expected_intent": "academic", "category": "definition"},
    {"query": "Nên học ngôn ngữ lập trình nào?", "expected_intent": "academic", "category": "it_advisory"},
    {"query": "Học CNTT cần những gì?", "expected_intent": "academic", "category": "it_general"},
    {"query": "Muốn tốt nghiệp cần bao nhiêu tín chỉ?", "expected_intent": "academic", "category": "graduation"},
    {"query": "Điều kiện miễn giảm học phí", "expected_intent": "academic", "category": "policy"},
    {"query": "Kỷ luật sinh viên quy định thế nào?", "expected_intent": "academic", "category": "discipline"},
]


# =============================================================================
# 4. CITATION ACCURACY TEST SET
#    Dùng cho anti-hallucination evaluation
# =============================================================================

CITATION_TESTS = [
    {
        "query": "Điều 43 quy định gì?",
        "must_cite_articles": ["43"],
        "must_not_cite_articles": ["36", "99"],
        "category": "specific",
        "description": "Phải trích dẫn đúng Điều 43",
    },
    {
        "query": "Nội dung Điều 36 là gì?",
        "must_cite_articles": ["36"],
        "must_not_cite_articles": ["43", "99"],
        "category": "specific",
        "description": "Phải trích dẫn đúng Điều 36",
    },
    {
        "query": "Điều 20 quy định cảnh báo học vụ thế nào?",
        "must_cite_articles": ["20"],
        "must_not_cite_articles": ["99"],
        "category": "specific",
        "description": "Phải trích dẫn Điều 20",
    },
    {
        "query": "Giảm hạng tốt nghiệp khi bị kỷ luật?",
        "must_cite_articles": ["43"],
        "must_not_cite_articles": [],
        "category": "crossref",
        "description": "Giảm hạng nằm ở Điều 43",
    },
    {
        "query": "Làm sao để bảo lưu kết quả?",
        "must_cite_articles": [],
        "must_not_cite_articles": ["99", "88", "77"],
        "category": "implicit",
        "description": "Query ngầm — không được bịa số Điều không tồn tại",
    },
]


# =============================================================================
# 5. PERFORMANCE TEST SET (Các query đa dạng để đo latency)
# =============================================================================

PERFORMANCE_QUERIES = [
    "Điều 43 quy định gì?",
    "Điều kiện cảnh báo học vụ là gì?",
    "Cần bao nhiêu tín chỉ để ra trường?",
    "Học lại và học cải thiện khác nhau thế nào?",
    "GPA bao nhiêu thì được học bổng?",
    "Xin chào",
    "Nên học ngôn ngữ lập trình nào?",
    "Điều 36 nói gì về điểm trung bình?",
    "Thời gian tối đa hoàn thành chương trình?",
    "Điều kiện miễn giảm học phí",
]
