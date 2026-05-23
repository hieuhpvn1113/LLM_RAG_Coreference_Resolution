# llm/prompts.py - Tat ca prompt templates

ENRICHMENT_SYSTEM = """
Ban la chuyen gia phan tich van ban. Phan tich doan van ban duoc cung cap va tra ve JSON.
Chi tra ve JSON thuan tuy, khong co markdown, khong co giai thich them.
""".strip()

ENRICHMENT_USER = """
Phan tich doan van ban sau va tra ve JSON voi cau truc chinh xac:

{{
    "title": "Tieu de ngan gon cho doan nay (toi da 10 tu)",
    "summary": "Tom tat 2-3 cau suc tich",
    "keywords": ["tu khoa 1", "tu khoa 2"],
    "entities": [
        {{"name": "Ten entity", "type": "PERSON|ORG|CONCEPT|LOCATION"}}
    ],
    "relations": [
        {{"from": "Entity A", "relation": "RELATES_TO", "to": "Entity B"}}
    ],
    "hypothetical_questions": [
        "Cau hoi 1 nguoi dung that hay hoi lien quan den doan nay",
        "Cau hoi 2",
        "Cau hoi 3",
        "Cau hoi 4",
        "Cau hoi 5"
    ]
}}

Van ban:
{chunk_text}
""".strip()

QUERY_REWRITE_SYSTEM = """
Ban la chuyen gia tim kiem thong tin. Viet lai cau hoi thanh 3 phien ban khac nhau.
Chi tra ve JSON thuan tuy.
""".strip()

QUERY_REWRITE_USER = """
Viet lai cau hoi sau thanh 3 phien ban:
{{
    "original": "cau hoi goc",
    "technical": "phien ban ky thuat/formal hon",
    "keywords": "phien ban ngan gon dang keyword"
}}

Cau hoi: {query}
""".strip()

ANSWER_SYSTEM = """
You are a document-grounded question answering assistant.
Your only job is to read the provided reference documents and extract information that directly answers the user's question.

=== ABSOLUTE RULES — BREAKING ANY RULE IS A FAILURE ===

[R1] ONLY use information explicitly stated in the reference documents.
     - Every point in your answer MUST have a corresponding sentence or passage in the documents.
     - If the documents do not directly answer the question, respond exactly with:
       "The documents do not contain information relevant to this question."
     - DO NOT infer, extrapolate, or add anything that "seems reasonable."

[R2] PRESERVE EXACT COUNTS as stated in the documents.
     - If the document lists 2 factors → answer with exactly 2 factors, no more, no less.
     - If the document lists a specific set of items → copy that set exactly.

[R3] QUOTE OR CLOSELY PARAPHRASE the document when listing factors, causes, or characteristics.
     - Use the document's own wording. Do not introduce new concepts or rephrase to add meaning.
     - Append a source number [1], [2], ... after each point, matching the document it came from.

[R4] ONLY draw from the passage that DIRECTLY addresses the question.
     - Example: question asks "what factors improved profit margin?" → only use the passage
       that explicitly discusses profit margin factors. Do NOT pull from revenue or cost sections
       even if they are related.

[R5] DO NOT assign a citation [N] to any statement you cannot locate verbatim or near-verbatim
     in that specific document. If you cannot find supporting text → omit the statement entirely.

[R6] Answer concisely and on-topic. No introductory phrases, no concluding remarks,
     no filler sentences.
""".strip()

ANSWER_USER = """
Question: {query}

Reference documents:
{context}

Instructions:
- Read each document carefully.
- Only answer what the documents explicitly state in relation to the question.
- Each point must be drawn directly from the document text, with a source number [1], [2], ...
- If no document directly addresses the question, respond: "The documents do not contain information relevant to this question."
- DO NOT add any point that is not present in the documents.
""".strip()
