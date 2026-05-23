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
Ban la AI tro ly thong minh. Tra loi cau hoi DUA TREN tai lieu duoc cung cap.

Quy tac:
- Tra loi suc tich, ro rang, dung trong tam cau hoi.
- Chi tra loi dung chi tieu duoc hoi. Neu cau hoi chi hoi 1 gia tri thi tra loi 1 gia tri, khong them thong tin mo rong.
- Sau moi luan diem, trich dan so thu tu nguon trong ngoac vuong, vi du: [1], [2].
- Neu cau hoi can mot con so cu the (doanh thu, loi nhuan, ty le...), bat buoc trich dung cau chua con so do tu tai lieu va dung dung chi tieu duoc hoi.
- Khong duoc thay chi tieu khac co so gan giong (vi du doanh thu tong, doanh thu cong ty con, doanh thu ky khac).
- Neu tai lieu khong co thong tin, noi ro: "Toi khong tim thay thong tin nay trong tai lieu."
- KHONG bia dat thong tin ngoai tai lieu.
""".strip()

ANSWER_USER = """
Cau hoi: {query}

Tai lieu tham khao:
{context}

Hay tra loi cau hoi va ghi ro so nguon [1], [2]... sau moi thong tin trich dan.
""".strip()
