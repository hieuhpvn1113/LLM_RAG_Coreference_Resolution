# llm/prompts.py — Tất cả prompt templates

ENRICHMENT_SYSTEM = """
Bạn là chuyên gia phân tích văn bản. Phân tích đoạn văn bản được cung cấp và trả về JSON.
Chỉ trả về JSON thuần túy, không có markdown, không có giải thích thêm.
""".strip()

ENRICHMENT_USER = """
Phân tích đoạn văn bản sau và trả về JSON với cấu trúc chính xác:

{{
    "title": "Tiêu đề ngắn gọn cho đoạn này (tối đa 10 từ)",
    "summary": "Tóm tắt 2-3 câu súc tích",
    "keywords": ["từ khóa 1", "từ khóa 2"],
    "entities": [
        {{"name": "Tên entity", "type": "PERSON|ORG|CONCEPT|LOCATION"}}
    ],
    "relations": [
        {{"from": "Entity A", "relation": "RELATES_TO", "to": "Entity B"}}
    ],
    "hypothetical_questions": [
        "Câu hỏi 1 người dùng thật hay hỏi liên quan đến đoạn này",
        "Câu hỏi 2",
        "Câu hỏi 3",
        "Câu hỏi 4",
        "Câu hỏi 5"
    ]
}}

Văn bản:
{chunk_text}
""".strip()

QUERY_REWRITE_SYSTEM = """
Bạn là chuyên gia tìm kiếm thông tin. Viết lại câu hỏi thành 3 phiên bản khác nhau.
Chỉ trả về JSON thuần túy.
""".strip()

QUERY_REWRITE_USER = """
Viết lại câu hỏi sau thành 3 phiên bản:
{{
    "original": "câu hỏi gốc",
    "technical": "phiên bản kỹ thuật/formal hơn",
    "keywords": "phiên bản ngắn gọn dạng keyword"
}}

Câu hỏi: {query}
""".strip()

ANSWER_SYSTEM = """
Bạn là AI trợ lý thông minh. Trả lời câu hỏi DỰA TRÊN tài liệu được cung cấp.

Quy tắc:
- Trả lời súc tích, rõ ràng, đúng trọng tâm câu hỏi.
- Sau mỗi luận điểm, trích dẫn số thứ tự nguồn trong ngoặc vuông, ví dụ: [1], [2].
- Nếu tài liệu không có thông tin, nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu."
- KHÔNG bịa đặt thông tin ngoài tài liệu.
""".strip()

ANSWER_USER = """
Câu hỏi: {query}

Tài liệu tham khảo:
{context}

Hãy trả lời câu hỏi và ghi rõ số nguồn [1], [2]... sau mỗi thông tin trích dẫn.
""".strip()
