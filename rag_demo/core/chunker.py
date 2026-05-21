# core/chunker.py — Semantic + Hierarchical Chunking
"""
Chiến lược 2 cấp:

  Level 1 (Section)
    • 1 chunk cha = 1 chương hoàn chỉnh
    • Flush CHỈ tại heading chính (Roman I./II./III. hoặc CHƯƠNG/PHẦN)
    • Không giới hạn token — mục 1, 2, 3… luôn thuộc cùng cha

  Level 2 (Paragraph)
    • Chia từng chương thành các chunk con ngữ nghĩa
    • Dùng embedding cosine-distance để tìm điểm cắt tự nhiên
    • Threshold ADAPTIVE: mean + 0.5*std của distances trong chính section đó
        → văn bản pháp luật (distances thấp đều) tự chỉnh ngưỡng xuống
        → văn bản kỹ thuật (distances cao khi chuyển section) tự chỉnh ngưỡng lên
    • WIN=3: mỗi điểm cắt so sánh trung bình 3 câu bên trái vs 3 câu bên phải
    • Khi 1 unit vượt CHUNK_SIZE_PARAGRAPH token → cắt theo ranh giới CÂU:
        gom câu vào chunk, khi token ≥ limit VÀ vừa kết thúc câu → flush
        → chunk có thể hơi vượt giới hạn, nhưng LUÔN kết thúc bằng câu hoàn chỉnh
    • Heading mục ("1. ...", "2. ...") giữ nguyên với nội dung theo sau
    • KHÔNG overlap giữa các chunk con
        → mỗi chunk con là đơn vị ngữ nghĩa độc lập, vector thuần, search chính xác hơn
        → đã có L1 parent chứa toàn bộ chương làm context khi cần

Thứ tự gọi:
    sections  = hierarchical_split(full_text, doc_id, source_file)
    l2_chunks = semantic_split(section, doc_id)
"""

import re
import uuid
import numpy as np
import tiktoken

from config import CHUNK_SIZE_PARAGRAPH

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def clean_text(text: str) -> str:
    text = text.replace('\ufeff', '')
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' \n', '\n', text)
    return text.strip()

def _safe_token_decode(token_ids: list) -> str:
    raw_bytes = _enc.decode_bytes(token_ids)
    return raw_bytes.decode('utf-8', errors='replace').lstrip('\ufffd').strip()


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    try:
        import nltk
        try:
            sents = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Sentence-boundary chunker
# ---------------------------------------------------------------------------
def _split_at_sentence_boundary(sentences: list[str], max_tokens: int) -> list[str]:
    """
    Gom câu vào chunk theo nguyên tắc:
      • Thêm câu vào chunk hiện tại
      • Khi tổng token ≥ max_tokens VÀ vừa kết thúc câu → flush, bắt đầu chunk mới
      • Câu đơn dài hơn max_tokens → vẫn giữ nguyên 1 chunk (không cắt giữa câu)

    Kết quả: mỗi chunk luôn kết thúc ở '.' (hoặc '!' '?') — không bao giờ bị cụt.
    """
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tok = 0

    for sent in sentences:
        sent_tok = count_tokens(sent)
        current.append(sent)
        current_tok += sent_tok

        if current_tok >= max_tokens:
            chunks.append(' '.join(current).strip())
            current = []
            current_tok = 0

    if current:
        chunks.append(' '.join(current).strip())

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Cosine distance
# ---------------------------------------------------------------------------
def _cosine_distance(v1: list, v2: list) -> float:
    a, b = np.array(v1), np.array(v2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Heading detectors
# ---------------------------------------------------------------------------
_MAJOR_SECTION_RE = re.compile(
    r'^\s*(?:'
    r'(?:I{1,3}|IV|VI{0,3}|IX|X{1,2}(?:I{0,3}|IV|VI{0,3}|IX)?|XI{0,3})\.\s+\S.{0,250}'
    r'|(?:Chương|CHƯƠNG|Phần|PHẦN|Mục|MỤC)\s+[\dIVXLCDM]+[.:\s]\s*\S.{0,250}'
    r')\s*$',
    re.IGNORECASE,
)

# Heading mục cấp 2: "1. ...", "2. ...", "a) ...", "b) ..."
_SUB_HEADING_RE = re.compile(
    r'^\s*(?:\d+\.|[a-zđ]\))\s+\S.{0,200}\s*$',
    re.IGNORECASE,
)

def _is_major_section_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 350 and bool(_MAJOR_SECTION_RE.match(line))

def _is_sub_heading(line: str) -> bool:
    """True cho heading mục cấp 2: '1. Cơ sở chính trị', '2. Quan điểm', 'a) ...'"""
    line = line.strip()
    return bool(line) and bool(_SUB_HEADING_RE.match(line))

_MINOR_HEADING_RE = re.compile(
    r'^\s*(?:'
    r'#{1,2}\s+.+'
    r'|(?:I{1,3}|IV|VI{0,3}|IX|X{0,3}(?:I{1,3}|IV|VI{0,3}|IX)?)\.\s+\S.{0,200}'
    r'|(?:Chương|CHƯƠNG|Phần|PHẦN|Mục|MỤC)\s+[\dIVXLCDM]+[.:\s]\s*\S.{0,200}'
    r')\s*$',
    re.IGNORECASE,
)

def _is_all_caps_title(text: str) -> bool:
    stripped = text.strip()
    if len(stripped.split()) < 4:
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) < 10:
        return False
    upper_letters = [ch for ch in letters if ch.isupper()]
    return len(upper_letters) / len(letters) >= 0.85 and len(stripped) <= 120

def _is_section_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 300:
        return False
    return bool(_MINOR_HEADING_RE.match(line)) or _is_all_caps_title(line)

def _normalize_to_blocks(text: str) -> list[str]:
    """
    Tách text thành blocks tại blank line hoặc ranh giới heading.
    Heading mục ("1. ...", "2. ...") KHÔNG tách thành block riêng —
    chúng sẽ được gom với nội dung theo sau trong _group_heading_with_content.
    """
    raw_paragraphs = re.split(r'\n\n+', text)
    blocks = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.split('\n')
        if len(lines) == 1:
            blocks.append(para)
            continue
        current_lines = []
        for line in lines:
            ls = line.strip()
            if not ls:
                continue
            # Chỉ tách heading CHÍNH và ALL_CAPS thành block riêng
            # Heading mục (1. / 2. / a)) giữ nguyên với nội dung
            if _is_section_heading(ls) and not _is_sub_heading(ls):
                if current_lines:
                    blocks.append('\n'.join(current_lines))
                    current_lines = []
                blocks.append(ls)
            else:
                current_lines.append(line)
        if current_lines:
            blocks.append('\n'.join(current_lines))
    return [b for b in blocks if b.strip()]


# ---------------------------------------------------------------------------
# STEP 1: Hierarchical Split → Level 1
# ---------------------------------------------------------------------------
def hierarchical_split(text: str, doc_id: str, source_file: str) -> list:
    """
    1 chương = 1 chunk cha.
    Flush CHỈ tại heading chính Roman numeral / CHƯƠNG / PHẦN.
    Không giới hạn token ở L1.
    """
    text = clean_text(text)
    blocks = _normalize_to_blocks(text)
    if not blocks:
        return []

    sections_raw: list[str] = []
    current_parts: list[str] = []

    for block in blocks:
        if _is_major_section_heading(block) and current_parts:
            sections_raw.append('\n\n'.join(current_parts))
            current_parts = []
        current_parts.append(block)

    if current_parts:
        sections_raw.append('\n\n'.join(current_parts))

    # Merge section quá nhỏ (< 80 token) vào trước — trừ khi là heading chính
    merged: list[str] = []
    for s in sections_raw:
        s = s.strip()
        first_line = s.split('\n')[0].strip()
        if merged and count_tokens(s) < 80 and not _is_major_section_heading(first_line):
            merged[-1] = merged[-1] + '\n\n' + s
        else:
            merged.append(s)

    chunks = []
    char_cursor = 0
    for idx, section_text in enumerate(merged):
        if not section_text.strip():
            continue
        cleaned = clean_text(section_text)
        pos = text.find(section_text[:80], char_cursor)
        char_start = pos if pos != -1 else char_cursor
        char_end   = char_start + len(section_text)
        char_cursor = char_end
        chunks.append({
            'chunk_id':    str(uuid.uuid4()),
            'doc_id':      doc_id,
            'level':       1,
            'parent_id':   None,
            'prev_id':     None,
            'next_id':     None,
            'seq_no':      str(idx),
            'raw_text':    section_text,
            'clean_text':  cleaned,
            'token_count': count_tokens(cleaned),
            'source_file': source_file,
            'char_start':  char_start,
            'char_end':    char_end,
        })

    if not chunks:
        cleaned = clean_text(text)
        chunks.append({
            'chunk_id':    str(uuid.uuid4()),
            'doc_id':      doc_id,
            'level':       1,
            'parent_id':   None,
            'prev_id':     None,
            'next_id':     None,
            'seq_no':      '0',
            'raw_text':    text,
            'clean_text':  cleaned,
            'token_count': count_tokens(cleaned),
            'source_file': source_file,
            'char_start':  0,
            'char_end':    len(text),
        })

    for i, c in enumerate(chunks):
        if i > 0:
            c['prev_id'] = chunks[i - 1]['chunk_id']
        if i < len(chunks) - 1:
            c['next_id'] = chunks[i + 1]['chunk_id']

    return chunks


# ---------------------------------------------------------------------------
# STEP 2: Semantic Split → Level 2  (no overlap, sentence-boundary cut)
# ---------------------------------------------------------------------------
def _semantic_units(text: str) -> list[str]:
    """
    Dùng embedding cosine-distance để tìm điểm cắt ngữ nghĩa tự nhiên.

    Thay đổi so với phiên bản cũ:
      • WIN=3 (tăng từ 2): so sánh trung bình 3 câu trái vs 3 câu phải
        → nhạy hơn với chuyển chủ đề thật, ít bị noise câu đơn lẻ
      • Adaptive threshold = mean + 0.5*std của tất cả distances trong section
        → tự chỉnh theo từng loại văn bản, không dùng ngưỡng cứng 0.25

    Heading mục ("1. ...", "2. ...") KHÔNG bị tách khỏi câu đầu tiên theo sau.
    """
    from core.embedder import embed_batch

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    vectors = embed_batch(sentences)

    # ── WIN=3: tính distances tại mọi điểm có thể cắt ──────────────────────
    WIN = 3
    scored: list[tuple[int, float]] = []
    for i in range(WIN, len(sentences) - WIN):
        v_left  = np.mean([vectors[j] for j in range(max(0, i - WIN), i)], axis=0)
        v_right = np.mean([vectors[j] for j in range(i, min(len(vectors), i + WIN))], axis=0)
        dist = _cosine_distance(v_left.tolist(), v_right.tolist())
        scored.append((i, dist))

    if not scored:
        return [text]

    # ── Adaptive threshold: mean + 0.5*std của section này ──────────────────
    dist_vals = np.array([d for _, d in scored])
    adaptive_threshold = float(np.mean(dist_vals) + 0.5 * np.std(dist_vals))

    print(f"[chunker]   adaptive_threshold={adaptive_threshold:.3f} "
          f"(mean={np.mean(dist_vals):.3f}, std={np.std(dist_vals):.3f}, "
          f"n_candidates={len(scored)})")

    # ── Xác định điểm cắt ───────────────────────────────────────────────────
    cut_indices: set[int] = {i for i, d in scored if d > adaptive_threshold}

    # Không cắt ngay SAU heading mục — heading phải đi cùng câu đầu của đoạn nó dẫn
    protected: set[int] = set()
    for i in cut_indices:
        if _is_sub_heading(sentences[i - 1]):
            protected.add(i)
    cut_indices -= protected

    # ── Tạo groups từ cut_indices ────────────────────────────────────────────
    groups: list[list[str]] = []
    start = 0
    for cut in sorted(cut_indices):
        if cut > start:
            groups.append(sentences[start:cut])
        start = cut
    if start < len(sentences):
        groups.append(sentences[start:])
    if not groups:
        groups = [sentences]

    # Merge nhóm quá ngắn (< 80 token) vào nhóm liền trước
    merged: list[list[str]] = []
    for g in groups:
        g_tok = count_tokens(' '.join(g))
        if merged and g_tok < 80:
            merged[-1] += g
        else:
            merged.append(list(g))

    return [' '.join(g).strip() for g in merged if g]


def semantic_split(section: dict, doc_id: str) -> list:
    """
    Chia L1 section thành L2 chunks:
      1. Dùng cosine-distance (adaptive threshold, WIN=3) để tạo semantic units
      2. Unit nào ≤ CHUNK_SIZE_PARAGRAPH → 1 chunk
      3. Unit nào > CHUNK_SIZE_PARAGRAPH → dùng _split_at_sentence_boundary:
            gom câu cho đến khi token ≥ limit VÀ vừa kết thúc câu → flush
            → chunk LUÔN kết thúc bằng câu hoàn chỉnh, bắt đầu bằng câu hoàn chỉnh
      Không overlap giữa các chunk.
    """
    text        = section['clean_text']
    parent_id   = section['chunk_id']
    source_file = section.get('source_file', '')
    parent_seq  = str(section['seq_no'])

    # Section ngắn → 1 chunk L2
    if count_tokens(text) <= CHUNK_SIZE_PARAGRAPH:
        return [{
            'chunk_id':    str(uuid.uuid4()),
            'doc_id':      doc_id,
            'level':       2,
            'parent_id':   parent_id,
            'prev_id':     None,
            'next_id':     None,
            'seq_no':      f"{parent_seq}.{0:04d}",
            'raw_text':    text,
            'clean_text':  text,
            'token_count': count_tokens(text),
            'source_file': source_file,
        }]

    print(f"[chunker] Semantic split section {parent_seq} ({count_tokens(text)} tokens)")

    # Bước 1: chia thành semantic units (adaptive threshold + WIN=3)
    units = _semantic_units(text)
    if not units:
        units = [text]

    # Bước 2: mỗi unit — giữ nguyên nếu ngắn, chia theo ranh giới câu nếu dài
    final_texts: list[str] = []
    for unit in units:
        if not unit:
            continue
        if count_tokens(unit) <= CHUNK_SIZE_PARAGRAPH:
            final_texts.append(unit)
        else:
            sents = _split_sentences(unit)
            sub_chunks = _split_at_sentence_boundary(sents, CHUNK_SIZE_PARAGRAPH)
            final_texts.extend(sub_chunks)

    # Bước 3: đóng gói thành chunk dict
    chunks = []
    for chunk_text in final_texts:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        chunks.append({
            'chunk_id':    str(uuid.uuid4()),
            'doc_id':      doc_id,
            'level':       2,
            'parent_id':   parent_id,
            'prev_id':     None,
            'next_id':     None,
            'seq_no':      f"{parent_seq}.{len(chunks):04d}",
            'raw_text':    chunk_text,
            'clean_text':  chunk_text,
            'token_count': count_tokens(chunk_text),
            'source_file': source_file,
        })

    for i, c in enumerate(chunks):
        if i > 0:
            c['prev_id'] = chunks[i - 1]['chunk_id']
        if i < len(chunks) - 1:
            c['next_id'] = chunks[i + 1]['chunk_id']

    return chunks
