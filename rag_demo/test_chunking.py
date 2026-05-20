"""
test_chunking.py — Kiểm tra nhanh kết quả hierarchical_split
Chạy: python rag_demo/test_chunking.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.chunker import hierarchical_split, clean_text

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "input.txt")

def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        raw = f.read()
    text = clean_text(raw)
    chunks = hierarchical_split(text, doc_id="test-doc", source_file="input.txt")

    print(f"\n{'='*60}")
    print(f"  Tổng số chunk cha (Level 1): {len(chunks)}")
    print(f"{'='*60}")
    for i, c in enumerate(chunks):
        first_line = c['clean_text'].split('\n')[0].strip()[:80]
        print(f"\n  Chunk #{i}  |  {c['token_count']:4d} tokens  |  seq={c['seq_no']}")
        print(f"  ┌─ {first_line}")
        if c['token_count'] > 30:
            last_line = c['clean_text'].strip().split('\n')[-1].strip()[:80]
            print(f"  └─ ...{last_line}")

    print(f"\n{'='*60}")
    print("  Kết quả mong đợi:")
    print("    Chunk #0  → header + TÀI LIỆU GIỚI THIỆU + mở đầu")
    print("    Chunk #1  → I. SỰ CẦN THIẾT BAN HÀNH LUẬT ĐƯỜNG BỘ")
    print("    Chunk #2  → II. MỤC ĐÍCH, QUAN ĐIỂM XÂY DỰNG LUẬT")
    print("    Chunk #3  → III. BỐ CỤC VÀ NHỮNG ĐIỂM MỚI...")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
