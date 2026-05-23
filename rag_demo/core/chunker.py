# core/chunker.py — Semantic + Hierarchical Chunking  (+ Coref Pre-processing)
"""
Chiến lược 2 cấp:

  Level 1 (Section)
    • 1 chunk cha = 1 phần/mục hoàn chỉnh theo cấu trúc tài liệu
    • Flush tại heading L1 theo thứ tự ưu tiên:
        1. Page-header (vd: "2 CTCP SỮA VIỆT NAM...")  ← loại bỏ, không giữ lại
        2. ALL-CAPS title  (vd: "TÓM TẮT KẾT QUẢ KINH DOANH...")
        3. Roman numeral / CHƯƠNG / PHẦN / MỤC
        4. Metric-heading  (vd: "Biên lợi nhuận: ...")
    • Không flush tại: i./ii./iii., a)/b), sub-heading thứ cấp
    • Không giới hạn token — mục 1, 2, 3… luôn thuộc cùng cha
    • Merge chunk L1 quá nhỏ (<80 token, không phải major heading)
      vào chunk trước để tránh L1 lùn

  Level 2 (Paragraph)  ← CÓ COREF PRE-PROCESSING
    • Trước semantic split: resolve_coref(L1.clean_text) → coref_text
        - raw_text của L1 KHÔNG thay đổi → LLM đọc văn gốc 100%
        - coref_text dùng để: (a) tìm điểm cắt chuẩn hơn, (b) L2 clean_text
    • Tách orig_sents và coref_sents (1:1 mapping câu)
    • Dùng coref_sents để embed → tìm cut_indices (cùng 1 bộ index)
    • Áp dụng cut_indices lên cả hai → L2 raw_text=orig, clean_text=coref
    • Threshold ADAPTIVE: mean + 0.5*std
    • WIN=3: mỗi điểm cắt so sánh 3 câu trái vs 3 câu phải
    • Unit vượt CHUNK_SIZE → cắt theo ranh giới câu (sentence-boundary)
    • KHÔNG overlap

  raw_text  → giữ nguyên (LLM đọc)
  clean_text → coref-resolved (embedding / Qdrant / ES / Neo4j)

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


def _split_sentences_paired(
    orig_sents: list[str],
    coref_sents: list[str],
    max_tokens: int,
) -> list[tuple[str, str]]:
    if not orig_sents:
        return []
    pairs: list[tuple[str, str]] = []
    cur_orig:  list[str] = []
    cur_coref: list[str] = []
    cur_tok = 0

    for o_sent, c_sent in zip(orig_sents, coref_sents):
        sent_tok = count_tokens(c_sent)
        cur_orig.append(o_sent)
        cur_coref.append(c_sent)
        cur_tok += sent_tok
        if cur_tok >= max_tokens:
            pairs.append((' '.join(cur_orig).strip(), ' '.join(cur_coref).strip()))
            cur_orig  = []
            cur_coref = []
            cur_tok   = 0

    if cur_orig:
        pairs.append((' '.join(cur_orig).strip(), ' '.join(cur_coref).strip()))

    return [(o, c) for o, c in pairs if o and c]


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

# Page header dạng: "2 CTCP SỮA VIỆT NAM (VNM) - BẢN TIN NHÀ ĐẦU TƯ QUÝ 1 NĂM 2026"
_PAGE_HEADER_RE = re.compile(
    r'^\s*\d+\s+CTCP\s+S[ỮƯ]A\s+VI[ỆE]T\s+NAM.*B[ẢA]N\s+TIN\s+NH[ÀA]\s+Đ[ẦA]U\s+T[ƯU].*$',
    re.IGNORECASE,
)

# Roman numeral / CHƯƠNG / PHẦN / MỤC (major structural headings)
_MAJOR_SECTION_RE = re.compile(
    r'^\s*(?:'
    r'(?:I{1,3}|IV|VI{0,3}|IX|X{1,2}(?:I{0,3}|IV|VI{0,3}|IX)?|XI{0,3})\.\s+\S.{0,250}'
    r'|(?:Chương|CHƯƠNG|Phần|PHẦN|Mục|MỤC)\s+[\dIVXLCDM]+[.:\s]\s*\S.{0,250}'
    r')\s*$',
    re.IGNORECASE,
)

# Sub-heading: i./ii./iii. hoặc a)/b)/c) — KHÔNG phải L1 boundary
_SUB_HEADING_RE = re.compile(
    r'^\s*(?:[ivxlcdmIVXLCDM]{1,4}\.|[a-zđA-ZĐ]\))\s+\S.{0,200}\s*$',
)

# Metric heading dạng "Biên lợi nhuận: ..."
_METRIC_HEADING_RE = re.compile(
    r'^\s*(?:'
    r'bi[eê]n l[ợo]i nhu[ậa]n'
    r'|doanh thu'
    r'|chi ph[ií] ho[ạa]t đ[ộo]ng'
    r'|l[ợo]i nhu[ậa]n(?:\s+sau thu[ếe])?'
    r'|t[iì]nh h[iì]nh t[aà]i ch[ií]nh'
    r'|k[eế]t qu[aả] ho[aạ]t đ[oộ]ng'
    r')\s*:\s*\S.*$',
    re.IGNORECASE,
)

# Minor heading cho normalize_to_blocks (L2 level)
_MINOR_HEADING_RE = re.compile(
    r'^\s*(?:'
    r'#{1,2}\s+.+'
    r'|(?:I{1,3}|IV|VI{0,3}|IX|X{0,3}(?:I{1,3}|IV|VI{0,3}|IX)?)\.\s+\S.{0,200}'
    r'|(?:Chương|CHƯƠNG|Phần|PHẦN|Mục|MỤC)\s+[\dIVXLCDM]+[.:\s]\s*\S.{0,200}'
    r')\s*$',
    re.IGNORECASE,
)


def _is_page_header(line: str) -> bool:
    """Nhận diện header trang lặp lại (vd: '2 CTCP SỮA VIỆT NAM...')."""
    return bool(_PAGE_HEADER_RE.match(line.strip()))


def _is_all_caps_title(text: str) -> bool:
    """
    Nhận diện tiêu đề ALL-CAPS:
    - Tối thiểu 3 từ, tối đa 180 ký tự
    - ≥ 85% chữ cái là uppercase
    - Không phải page-header
    - Có thể kết thúc bằng dấu ':' (vd: "TÓM TẮT KẾT QUẢ KINH DOANH:")
    """
    stripped = text.strip()
    # Loại page-header
    if _is_page_header(stripped):
        return False
    if len(stripped) > 180 or len(stripped.split()) < 3:
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) < 8:
        return False
    upper_letters = [ch for ch in letters if ch.isupper()]
    return len(upper_letters) / len(letters) >= 0.85


def _is_major_section_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 350 and bool(_MAJOR_SECTION_RE.match(line))


def _is_sub_heading(line: str) -> bool:
    """Sub-heading dạng i./ii. hoặc a)/b) — KHÔNG flush L1."""
    line = line.strip()
    return bool(line) and bool(_SUB_HEADING_RE.match(line))


def _is_section_heading(line: str) -> bool:
    """Dùng nội bộ trong _normalize_to_blocks (L2 level)."""
    line = line.strip()
    if not line or len(line) > 300:
        return False
    return (
        bool(_MINOR_HEADING_RE.match(line))
        or bool(_METRIC_HEADING_RE.match(line))
        or _is_all_caps_title(line)
    )


def _is_l1_boundary_heading(line: str) -> bool:
    """
    Xác định ranh giới flush L1.

    Thứ tự ưu tiên (từ cao đến thấp):
      1. Page-header  → True  (sẽ bị lọc khỏi nội dung sau)
      2. ALL-CAPS title (≥3 từ, ≥85% uppercase, ≤180 ký tự)
      3. Roman numeral / CHƯƠNG / PHẦN / MỤC heading
      4. Metric-heading dạng "Tên mục: nội dung"

    KHÔNG flush tại:
      - Sub-heading kiểu i./ii./iii. hoặc a)/b)
      - Dòng thường có chữ thường xen lẫn
    """
    ls = (line or "").strip()
    if not ls:
        return False

    # Sub-heading → KHÔNG flush L1
    if _is_sub_heading(ls):
        return False

    return (
        _is_page_header(ls)
        or _is_all_caps_title(ls)
        or _is_major_section_heading(ls)
        or bool(_METRIC_HEADING_RE.match(ls))
    )


def _normalize_to_blocks(text: str) -> list[str]:
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
    Tách văn bản thành L1 chunks theo cấu trúc tài liệu.

    Thứ tự ưu tiên flush (từ cao đến thấp):
      1. ALL-CAPS title  (ưu tiên cao nhất — phổ biến trong báo cáo VN)
      2. Roman / CHƯƠNG / PHẦN
      3. Metric-heading
      4. Page-header     (bị bỏ qua, không đưa vào nội dung)

    Sub-heading i./ii./iii. KHÔNG tạo L1 mới.
    Không giới hạn token ở L1.
    Merge L1 quá nhỏ (<80 token, không phải major/all-caps heading) vào chunk trước.
    """
    text = clean_text(text)
    blocks = _normalize_to_blocks(text)
    if not blocks:
        return []

    sections_raw: list[str] = []
    current_parts: list[str] = []

    for block in blocks:
        ls = block.strip()

        # Bỏ qua page-header hoàn toàn (không đưa vào nội dung)
        if _is_page_header(ls):
            continue

        if _is_l1_boundary_heading(ls) and current_parts:
            sections_raw.append('\n\n'.join(current_parts))
            current_parts = []

        current_parts.append(block)

    if current_parts:
        sections_raw.append('\n\n'.join(current_parts))

    # Merge L1 lùn (<80 token) vào chunk trước
    # NHƯNG giữ nguyên nếu dòng đầu là ALL-CAPS title hoặc major heading
    merged: list[str] = []
    for s in sections_raw:
        s = s.strip()
        if not s:
            continue
        first_line = s.split('\n')[0].strip()
        is_strong_boundary = (
            _is_all_caps_title(first_line)
            or _is_major_section_heading(first_line)
        )
        if merged and count_tokens(s) < 80 and not is_strong_boundary:
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

        first_line = section_text.split('\n')[0].strip()
        print(f"[chunker] L1[{idx}] heading='{first_line[:60]}' "
              f"tokens={count_tokens(cleaned)}")

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
# STEP 2a: Tìm cut indices từ danh sách câu (core semantic logic)
# ---------------------------------------------------------------------------
def _find_semantic_cut_indices(sentences: list[str]) -> set[int]:
    """
    Embed sentences → tính cosine distance → adaptive threshold → trả về set cut indices.

    WIN=3: so sánh trung bình 3 câu trái vs 3 câu phải.
    Adaptive threshold = mean + 0.5*std của tất cả distances trong section.
    Protected: không cắt ngay SAU sub-heading.
    """
    from core.embedder import embed_batch

    if len(sentences) <= 1:
        return set()

    vectors = embed_batch(sentences)

    WIN = 3
    scored: list[tuple[int, float]] = []
    for i in range(WIN, len(sentences) - WIN):
        v_left  = np.mean([vectors[j] for j in range(max(0, i - WIN), i)],        axis=0)
        v_right = np.mean([vectors[j] for j in range(i, min(len(vectors), i + WIN))], axis=0)
        dist = _cosine_distance(v_left.tolist(), v_right.tolist())
        scored.append((i, dist))

    if not scored:
        return set()

    dist_vals = np.array([d for _, d in scored])
    adaptive_threshold = float(np.mean(dist_vals) + 0.5 * np.std(dist_vals))

    print(f"[chunker]   adaptive_threshold={adaptive_threshold:.3f} "
          f"(mean={np.mean(dist_vals):.3f}, std={np.std(dist_vals):.3f}, "
          f"n_candidates={len(scored)})")

    cut_indices: set[int] = {i for i, d in scored if d > adaptive_threshold}

    # Bảo vệ: không cắt ngay sau sub-heading
    protected: set[int] = {i for i in cut_indices if _is_sub_heading(sentences[i - 1])}
    cut_indices -= protected

    return cut_indices


def _apply_cut_indices_paired(
    orig_sents:  list[str],
    coref_sents: list[str],
    cut_indices: set[int],
) -> list[tuple[list[str], list[str]]]:
    groups_orig:  list[list[str]] = []
    groups_coref: list[list[str]] = []
    start = 0

    for cut in sorted(cut_indices):
        if cut > start:
            groups_orig.append(orig_sents[start:cut])
            groups_coref.append(coref_sents[start:cut])
        start = cut
    if start < len(orig_sents):
        groups_orig.append(orig_sents[start:])
        groups_coref.append(coref_sents[start:])

    if not groups_orig:
        return [(orig_sents, coref_sents)]

    merged_o: list[list[str]] = []
    merged_c: list[list[str]] = []
    for go, gc in zip(groups_orig, groups_coref):
        gc_tok = count_tokens(' '.join(gc))
        if merged_c and gc_tok < 80:
            merged_o[-1] += go
            merged_c[-1] += gc
        else:
            merged_o.append(list(go))
            merged_c.append(list(gc))

    return list(zip(merged_o, merged_c))


# ---------------------------------------------------------------------------
# STEP 2b: _semantic_units — wrapper nội bộ
# ---------------------------------------------------------------------------
def _semantic_units(text: str) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    cut_indices = _find_semantic_cut_indices(sentences)
    pairs = _apply_cut_indices_paired(sentences, sentences, cut_indices)
    return [' '.join(g).strip() for g, _ in pairs if g]


# ---------------------------------------------------------------------------
# STEP 2c: semantic_split — Level 2 với Coref Pre-processing
# ---------------------------------------------------------------------------
def semantic_split(section: dict, doc_id: str) -> list:
    """
    Chia L1 section thành L2 chunks với Coref Pre-processing.

      1. Lấy L1 clean_text (văn gốc đã clean, chưa coref)
      2. resolve_coref(clean_text) → coref_text
      3. Split cả hai thành sentences (1:1 mapping)
         Nếu số câu khác nhau → fallback coref_sents = orig_sents
      4. Dùng coref_sents để embed + tìm cut_indices
      5. Áp dụng cut_indices lên cả orig và coref
         → L2 raw_text  = orig segment  (văn gốc)
         → L2 clean_text = coref segment (đã resolve)
      6. Sub-split nếu unit vượt CHUNK_SIZE_PARAGRAPH
    """
    orig_text   = section['clean_text']
    parent_id   = section['chunk_id']
    source_file = section.get('source_file', '')
    parent_seq  = str(section['seq_no'])

    # Section ngắn → 1 chunk L2
    if count_tokens(orig_text) <= CHUNK_SIZE_PARAGRAPH:
        from core.coref_resolver import resolve_coref
        coref_text = resolve_coref(orig_text)
        return [{
            'chunk_id':    str(uuid.uuid4()),
            'doc_id':      doc_id,
            'level':       2,
            'parent_id':   parent_id,
            'prev_id':     None,
            'next_id':     None,
            'seq_no':      f"{parent_seq}.{0:04d}",
            'raw_text':    orig_text,
            'clean_text':  coref_text,
            'token_count': count_tokens(coref_text),
            'source_file': source_file,
        }]

    print(f"[chunker] Semantic split section {parent_seq} ({count_tokens(orig_text)} tokens)")

    from core.coref_resolver import resolve_coref
    coref_text = resolve_coref(orig_text)

    orig_sents  = _split_sentences(orig_text)
    coref_sents = _split_sentences(coref_text)

    if len(orig_sents) != len(coref_sents):
        print(f"[chunker] ⚠️  Coref changed sentence count "
              f"({len(orig_sents)} → {len(coref_sents)}), using orig for pairing")
        coref_sents = orig_sents

    cut_indices = _find_semantic_cut_indices(coref_sents)
    paired_groups = _apply_cut_indices_paired(orig_sents, coref_sents, cut_indices)

    final_pairs: list[tuple[str, str]] = []

    for orig_group, coref_group in paired_groups:
        orig_unit  = ' '.join(orig_group).strip()
        coref_unit = ' '.join(coref_group).strip()

        if not orig_unit:
            continue

        if count_tokens(coref_unit) <= CHUNK_SIZE_PARAGRAPH:
            final_pairs.append((orig_unit, coref_unit))
        else:
            sub_pairs = _split_sentences_paired(orig_group, coref_group, CHUNK_SIZE_PARAGRAPH)
            final_pairs.extend(sub_pairs)

    chunks = []
    for orig_chunk, coref_chunk in final_pairs:
        orig_chunk  = orig_chunk.strip()
        coref_chunk = coref_chunk.strip()
        if not orig_chunk:
            continue
        chunks.append({
            'chunk_id':    str(uuid.uuid4()),
            'doc_id':      doc_id,
            'level':       2,
            'parent_id':   parent_id,
            'prev_id':     None,
            'next_id':     None,
            'seq_no':      f"{parent_seq}.{len(chunks):04d}",
            'raw_text':    orig_chunk,
            'clean_text':  coref_chunk,
            'token_count': count_tokens(coref_chunk),
            'source_file': source_file,
        })

    for i, c in enumerate(chunks):
        if i > 0:
            c['prev_id'] = chunks[i - 1]['chunk_id']
        if i < len(chunks) - 1:
            c['next_id'] = chunks[i + 1]['chunk_id']

    return chunks
