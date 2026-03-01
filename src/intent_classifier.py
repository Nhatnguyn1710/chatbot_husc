"""Intent Classifier Module for HUSC Chatbot"""

import re
import random
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List


class Intent(Enum):
    SMALL_TALK = "small_talk"
    OUT_OF_SCOPE = "out_of_scope"
    ACADEMIC = "academic"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    response: Optional[str] = None
    matched_pattern: Optional[str] = None


class IntentClassifier:
    SMALL_TALK_PATTERNS: Dict[str, Dict] = {
        "greeting": {
            "patterns": [r"^(xin\s*)?chào(\s+bạn)?[.!]?$", r"^hello[.!]?$", r"^hi[.!]?$", r"^hey[.!]?$"],
            "responses": ["Xin chào! 👋 Tôi là trợ lý học vụ HUSC. Bạn cần mình hỗ trợ và giải đáp gì về quy chế học vụ , điểm số, hoặc thủ tục học vụ?"]
        },
        "thanks": {
            "patterns": [r"^(cảm\s*ơn|cám\s*ơn|thanks?|tks|thx)(\s+bạn)?[.!]?$"],
            "responses": ["Không có gì! 😊 Nếu còn thắc mắc gì, cứ hỏi nhé!"]
        },
        "goodbye": {
            "patterns": [r"^(tạm\s*biệt|bye|goodbye|bb)[.!]?$"],
            "responses": ["Tạm biệt và hẹn gặp lại! 👋 Chúc bạn học tập tốt!"]
        },
        "bot_identity": {
            "patterns": [r"^(bạn|mày)\s*(là\s*)?(ai|gì)[?]?$", r"^chatbot\s*(là\s*)?(gì|ai)[?]?$"],
            "responses": ["Mình là **Trợ lý học vụ HUSC** 🎓\n\nMình giúp bạn:\n- Tra cứu quy chế đào tạo\n- Tính điểm GPA\n- Giải đáp thủ tục học vụ\n\nHãy hỏi về học vụ nhé!"]
        },
        "how_are_you": {
            "patterns": [r"^(bạn|mày)\s*(có\s*)?(khỏe|ổn)[?]?$"],
            "responses": ["Mình ổn! 😊 Bạn cần hỏi gì về học vụ không?"]
        },
        "acknowledgment": {
            "patterns": [r"^(ok|oke|okay|ừ|ừm|được|đc)[.!]?$"],
            "responses": ["Oke! Còn thắc mắc gì về học vụ, cứ mình hỏi nhé!"]
        },
    }
    
    OUT_OF_SCOPE_PATTERNS: Dict[str, Dict] = {
        "weather": {
            "patterns": [r"(thời\s*tiết|trời|mưa|nắng|weather)"],
            "response": "Mình là trợ lý học vụ nên không có thông tin về thời tiết. 🌤️\n\nBạn cần hỏi gì về học vụ không?"
        },
        "entertainment": {
            "patterns": [r"(phim|movie|ca\s*sĩ|game|bóng\s*đá|thể\s*thao)"],
            "response": "Mình chuyên về học vụ HUSC nên không có thông tin giải trí. 📰\n\nBạn có câu hỏi về học vụ không?"
        },
        "food": {
            "patterns": [r"(ăn\s*gì|quán\s*ăn|nhà\s*hàng|cà\s*phê)"],
            "response": "Mình không có thông tin về ẩm thực. 🍜\n\nBạn cần hỏi gì về học vụ không?"
        },
        "emotional": {
            "patterns": [r"(yêu|thích|crush|tâm\s*sự|buồn)(?!.*môn)"],
            "response": "Mình chỉ là trợ lý học vụ thôi. 💭\n\nVề học vụ, mình sẵn sàng giúp bạn!"
        },
    }
    
    ACADEMIC_KEYWORDS: List[str] = [
        "điểm", "gpa", "tín chỉ", "học phần", "môn", "quy chế", "điều", "khoản", "chương",
        "học lại", "thi lại", "bảo lưu", "tốt nghiệp", "học bổng", "học phí",
        "đăng ký", "sinh viên", "husc", "trường", "khoa", "ngành"
    ]

    def __init__(self):
        self._compiled_small_talk: Dict = {}
        self._compiled_out_of_scope: Dict = {}
        self._academic_pattern: Optional[re.Pattern] = None
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        for cat, data in self.SMALL_TALK_PATTERNS.items():
            self._compiled_small_talk[cat] = {
                "patterns": [re.compile(p, re.IGNORECASE | re.UNICODE) for p in data["patterns"]],
                "responses": data["responses"]
            }
        for cat, data in self.OUT_OF_SCOPE_PATTERNS.items():
            self._compiled_out_of_scope[cat] = {
                "patterns": [re.compile(p, re.IGNORECASE | re.UNICODE) for p in data["patterns"]],
                "response": data["response"]
            }
        kw_pattern = r"\b(" + "|".join(re.escape(k) for k in self.ACADEMIC_KEYWORDS) + r")\b"
        self._academic_pattern = re.compile(kw_pattern, re.IGNORECASE | re.UNICODE)

    def classify(self, query: str) -> IntentResult:
        if not query or not query.strip():
            return IntentResult(Intent.SMALL_TALK, 1.0, "Xin chào! Bạn cần hỏi gì về học vụ?", "empty")
        
        query_clean = query.strip()
        
        # Check small talk
        for cat, data in self._compiled_small_talk.items():
            for p in data["patterns"]:
                if p.search(query_clean):
                    return IntentResult(Intent.SMALL_TALK, 0.95, random.choice(data["responses"]), f"small_talk:{cat}")
        
        # Check academic keywords first
        if self._academic_pattern and self._academic_pattern.search(query_clean.lower()):
            return IntentResult(Intent.ACADEMIC, 0.9, None, "academic_keywords")
        
        # Check out of scope
        for cat, data in self._compiled_out_of_scope.items():
            for p in data["patterns"]:
                if p.search(query_clean.lower()):
                    return IntentResult(Intent.OUT_OF_SCOPE, 0.85, data["response"], f"out_of_scope:{cat}")
        
        return IntentResult(Intent.ACADEMIC, 0.7, None, "default")


_classifier: Optional[IntentClassifier] = None

def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier

def classify_intent(query: str) -> IntentResult:
    return get_classifier().classify(query)
