# core/coref_resolver.py — Coreference Resolution cho văn bản pháp luật tiếng Việt
"""
Giải quyết tham chiếu mơ hồ (coref) TRƯỚC khi Semantic Split.

Mục tiêu:
  • clean_text của L1 được resolve → đưa vào _semantic_units() để cắt chuẩn hơn
  • L2 clean_text kế thừa text đã resolve → embedding / Qdrant / ES / Neo4j chính xác hơn
  • raw_text (L1) KHÔNG thay đổi → LLM vẫn đọc văn gốc 100%

Hai tầng:
  Tier 1 — Rule-based (luôn chạy, không cần model, nhanh):
    Phát hiện entity pháp luật (Điều X, Khoản X, Nghị định...) và
    thay thế đại từ / chỉ định từ bằng antecedent cụ thể.

  Tier 2 — Neural / fastcoref (tùy chọn, bật bằng COREF_MODE=neural):
    Dùng mô hình multilingual để xử lý các trường hợp phức tạp hơn.

Flow:
  L1 clean_text
      ↓ resolve_coref(text)
  coref_text (đã expand)
      ↓ semantic_split()
  L2 chunks (clean_text = coref_text segments)
"""

import re
from typing import Optional
from config import COREF_ENABLED, COREF_MODE


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Rule-based resolver (Vietnamese legal text)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Regex nhận diện entity pháp luật ──────────────────────────────────────────

# Văn bản quy phạm pháp luật (cụ thể nhất → match trước)
_RE_VAN_BAN = re.compile(
    r'(?:'
    r'Nghị định(?:\s+số)?\s+\d+/\d{4}/NĐ-CP'
    r'|Thông tư(?:\s+số)?\s+\d+/\d{4}/TT-\w+'
    r'|Quyết định(?:\s+số)?\s+\d+/\d{4}/QĐ-\w+'
    r'|Quyết định(?:\s+số)?\s+\d+'
    r'|Chỉ thị(?:\s+số)?\s+\d+/\d{4}/CT-\w+'
    r'|Nghị quyết(?:\s+số)?\s+\d+/\d{4}/NQ-\w+'
    r'|Luật\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂẮẶẦẤẬ][^\n,;.(]{3,60}'
    r')',
    re.IGNORECASE | re.UNICODE,
)

# Cấu trúc nội tại văn bản
_RE_DIEU   = re.compile(r'[Đđ]iều\s+\d+[a-z]?(?:\s+(?:Luật|Nghị định|Thông tư)[^\n,;.]{0,40})?')
_RE_KHOAN  = re.compile(r'[Kk]hoản\s+\d+(?:\s+[Đđ]iều\s+\d+[a-z]?)?')
_RE_DIEM   = re.compile(r'[Đđ]iểm\s+[a-zđ](?:\s+[Kk]hoản\s+\d+)?')
_RE_CHUONG = re.compile(r'[Cc]hương\s+(?:[IVXLCDM]+|\d+)')
_RE_MUC    = re.compile(r'[Mm]ục\s+(?:[IVXLCDM]+|\d+|[A-Z])')

# Chủ thể (tổ chức, cá nhân...)
_RE_CHU_THE = re.compile(
    r'(?:tổ chức|doanh nghiệp|cá nhân|hộ kinh doanh'
    r'|cơ quan nhà nước|đơn vị sự nghiệp|người lao động'
    r'|người sử dụng lao động|chủ đầu tư|nhà thầu)'
    r'(?:\s+[^\n,;.]{0,30})?',
    re.IGNORECASE | re.UNICODE,
)

# ── Đại từ / chỉ định từ cần resolve ─────────────────────────────────────────
# Dùng (?<!\w) / (?!\w) thay vì \b để tránh false match với ký tự Unicode VN

_WB_L = r'(?<!\w)'   # left word boundary — không dùng \b (sai với Unicode VN)
_WB_R = r'(?!\w)'    # right word boundary

_COREF_REFS: list[tuple[re.Pattern, str]] = [
    # "Điều này", "khoản đó", "mục trên" — cấu trúc văn bản + chỉ định từ
    (re.compile(
        _WB_L
        + r'([Đđ]iều|[Kk]hoản|[Đđ]iểm|[Mm]ục|[Cc]hương)'
        + r'\s+(này|đó|kia|trên|nêu trên|đã nêu|đã đề cập)'
        + _WB_R,
        re.UNICODE,
    ), 'struct'),

    # "quy định này/đó/trên"
    (re.compile(
        _WB_L + r'quy định\s+(này|đó|trên|nêu trên|đã nêu)' + _WB_R,
        re.UNICODE | re.IGNORECASE,
    ), 'regulation'),

    # "nội dung này/trên"
    (re.compile(
        _WB_L + r'nội dung\s+(này|đó|trên)' + _WB_R,
        re.UNICODE | re.IGNORECASE,
    ), 'content'),

    # "điều khoản này/trên"
    (re.compile(
        _WB_L + r'điều khoản\s+(này|đó|trên)' + _WB_R,
        re.UNICODE | re.IGNORECASE,
    ), 'clause'),

    # "văn bản này/nêu trên"
    (re.compile(
        _WB_L + r'văn bản\s+(này|đó|trên|nêu trên)' + _WB_R,
        re.UNICODE | re.IGNORECASE,
    ), 'van_ban'),

    # "nêu trên" / "đã nêu ở trên" đứng độc lập
    (re.compile(
        _WB_L + r'(nêu trên|đã nêu ở trên|đã đề cập ở trên)' + _WB_R,
        re.UNICODE | re.IGNORECASE,
    ), 'any'),

    # Đại từ nhân xưng chỉ chủ thể
    (re.compile(
        _WB_L + r'(họ|chúng|các bên)' + _WB_R,
        re.UNICODE | re.IGNORECASE,
    ), 'chu_the'),
]

# Regex kiểm tra matched_text ĐÃ chứa entity cụ thể (có số → đã rõ rồi, skip)
# Chỉ check digit thật — KHÔNG dùng [a-z]\b (sai với Unicode VN)
_RE_HAS_SPECIFIC = re.compile(r'\d+', re.UNICODE)


# ── Entity tracker ────────────────────────────────────────────────────────────

class _EntityTracker:
    """
    Theo dõi entity pháp luật xuất hiện gần nhất qua từng câu.
    """
    __slots__ = ('dieu', 'khoan', 'diem', 'chuong', 'muc', 'van_ban', 'chu_the', 'last_any')

    def __init__(self):
        self.dieu     = None
        self.khoan    = None
        self.diem     = None
        self.chuong   = None
        self.muc      = None
        self.van_ban  = None
        self.chu_the  = None
        self.last_any = None

    def update(self, sentence: str):
        """Cập nhật tracker từ các entity xuất hiện trong câu."""
        m = _RE_VAN_BAN.search(sentence)
        if m:
            self.van_ban  = m.group(0).strip()
            self.last_any = self.van_ban

        m = _RE_DIEU.search(sentence)
        if m:
            self.dieu     = m.group(0).strip()
            self.last_any = self.dieu

        m = _RE_KHOAN.search(sentence)
        if m:
            self.khoan    = m.group(0).strip()
            self.last_any = self.khoan

        m = _RE_DIEM.search(sentence)
        if m:
            self.diem     = m.group(0).strip()
            self.last_any = self.diem

        m = _RE_CHUONG.search(sentence)
        if m:
            self.chuong   = m.group(0).strip()
            self.last_any = self.chuong

        m = _RE_MUC.search(sentence)
        if m:
            self.muc      = m.group(0).strip()
            self.last_any = self.muc

        m = _RE_CHU_THE.search(sentence)
        if m:
            self.chu_the  = m.group(0).strip()

    def resolve(self, ref_type: str) -> Optional[str]:
        if ref_type == 'struct':
            return self.dieu or self.khoan or self.last_any
        elif ref_type == 'regulation':
            return self.dieu or self.khoan or self.van_ban or self.last_any
        elif ref_type in ('content', 'clause'):
            return self.last_any
        elif ref_type == 'van_ban':
            return self.van_ban or self.last_any
        elif ref_type == 'any':
            return self.last_any
        elif ref_type == 'chu_the':
            return self.chu_the
        return None


# ── Core resolve logic ────────────────────────────────────────────────────────

def _resolve_sentence(sentence: str, tracker: _EntityTracker) -> str:
    """
    Thay thế tham chiếu mơ hồ bằng antecedent từ tracker.

    Lưu ý thứ tự gọi trong resolve_coref_rules():
      → _resolve_sentence() TRƯỚC tracker.update()
      → đảm bảo "nêu trên" trong câu hiện tại dùng entity từ câu TRƯỚC,
        không bị ghi đè bởi entity trong chính câu này.
    """
    result = sentence

    for pattern, ref_type in _COREF_REFS:
        match = pattern.search(result)
        if not match:
            continue

        antecedent = tracker.resolve(ref_type)
        if not antecedent:
            continue

        matched_text = match.group(0)

        # Skip nếu matched text đã chứa số cụ thể (VD: "Điều 5 này" — đã rõ rồi)
        # CHỈ check digit thật, KHÔNG dùng [a-z]\b (false-positive với Unicode VN)
        if _RE_HAS_SPECIFIC.search(matched_text):
            continue

        # Skip nếu antecedent đã có trong matched (tránh thay thế vòng)
        if antecedent.lower() in matched_text.lower():
            continue

        replacement = _build_replacement(matched_text, antecedent, ref_type)
        result = result[:match.start()] + replacement + result[match.end():]

        # Chỉ resolve 1 lần mỗi pattern/câu
        break

    return result


def _build_replacement(matched: str, antecedent: str, ref_type: str) -> str:
    """
    Xây dựng chuỗi thay thế tự nhiên.

    Ví dụ:
      "Điều này"      + "Điều 5"                   → "Điều 5"
      "quy định trên" + "Điều 15"                  → "quy định tại Điều 15"
      "nêu trên"      + "Luật Giao thông đường bộ" → "nêu tại Luật Giao thông đường bộ"
      "nội dung này"  + "Điều 5"                   → "nội dung về Điều 5"
    """
    _DEMO = r'\s*(này|đó|kia|trên|nêu trên|đã nêu|đã đề cập)\s*$'

    if ref_type == 'struct':
        return antecedent

    elif ref_type in ('regulation', 'clause'):
        base = re.sub(_DEMO, '', matched, flags=re.IGNORECASE | re.UNICODE).strip()
        return f"{base} tại {antecedent}"

    elif ref_type == 'content':
        base = re.sub(_DEMO, '', matched, flags=re.IGNORECASE | re.UNICODE).strip()
        return f"{base} về {antecedent}"

    elif ref_type == 'van_ban':
        base = re.sub(_DEMO, '', matched, flags=re.IGNORECASE | re.UNICODE).strip()
        return f"{base} {antecedent}"

    elif ref_type == 'any':
        # "nêu trên" → "nêu tại <antecedent>"
        return f"nêu tại {antecedent}"

    elif ref_type == 'chu_the':
        return antecedent

    return matched


# ── Public API: Tier 1 ────────────────────────────────────────────────────────

def resolve_coref_rules(text: str) -> str:
    """
    Tier 1: Rule-based coreference resolution cho văn bản pháp luật VN.

    Thứ tự xử lý mỗi câu:
      1. _resolve_sentence() — dùng entity từ các câu TRƯỚC (tracker hiện tại)
      2. tracker.update()   — cập nhật entity từ câu này cho các câu SAU

    Tại sao resolve TRƯỚC update:
      "Những vấn đề nêu trên đặt ra yêu cầu ... Luật Đường bộ"
      → "nêu trên" phải trỏ về issues từ câu trước, không phải "Luật Đường bộ"
         trong cùng câu này.
    """
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    if not sentences:
        return text

    tracker = _EntityTracker()
    resolved = []

    for sent in sentences:
        # ── FIX: resolve TRƯỚC, update SAU ──────────────────────────────────
        resolved_sent = _resolve_sentence(sent, tracker)   # dùng entity câu trước
        tracker.update(sent)                                # cập nhật cho câu sau
        resolved.append(resolved_sent)

    return ' '.join(resolved)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Neural resolver (fastcoref, optional)
# ═══════════════════════════════════════════════════════════════════════════════

_neural_model  = None
_neural_loaded = False


def _load_neural_model():
    global _neural_model, _neural_loaded
    if _neural_loaded:
        return _neural_model
    _neural_loaded = True
    try:
        from fastcoref import FCoref
        _neural_model = FCoref(device='cpu')
        print("[coref] ✅ fastcoref model loaded (Tier 2)")
    except ImportError:
        print("[coref] ⚠️  fastcoref chưa cài — chỉ dùng Tier 1 (rule-based).")
        print("         Để bật Tier 2: pip install fastcoref")
        _neural_model = None
    except Exception as e:
        print(f"[coref] ⚠️  Không load được fastcoref: {e}")
        _neural_model = None
    return _neural_model


def resolve_coref_neural(text: str) -> str:
    model = _load_neural_model()
    if model is None:
        return resolve_coref_rules(text)
    try:
        preds    = model.predict(texts=[text])
        clusters = preds[0].get_clusters(as_strings=True)
        resolved = text
        replacements = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            representative = cluster[0]
            for mention in cluster[1:]:
                if mention != representative and len(mention) < len(representative):
                    replacements.append((mention, representative))
        replacements.sort(key=lambda x: -len(x[0]))
        for mention, rep in replacements:
            resolved = resolved.replace(mention, rep, 1)
        return resolved
    except Exception as e:
        print(f"[coref] Neural resolve lỗi: {e} — fallback rule-based")
        return resolve_coref_rules(text)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_coref(text: str) -> str:
    """
    Entry point chính. Điều phối Tier 1 / Tier 2 theo config.

    COREF_ENABLED=false  → bypass (trả text gốc)
    COREF_MODE=rule      → chỉ Tier 1 rule-based (default)
    COREF_MODE=neural    → chỉ Tier 2 fastcoref
    COREF_MODE=both      → Tier 1 trước, Tier 2 sau
    """
    if not COREF_ENABLED:
        return text
    if not text or not text.strip():
        return text

    mode = COREF_MODE.lower()

    if mode == 'rule':
        resolved = resolve_coref_rules(text)
    elif mode == 'neural':
        resolved = resolve_coref_neural(text)
    elif mode == 'both':
        resolved = resolve_coref_neural(resolve_coref_rules(text))
    else:
        print(f"[coref] ⚠️  COREF_MODE='{mode}' không hợp lệ — dùng 'rule'.")
        resolved = resolve_coref_rules(text)

    _log_changes(text, resolved)
    return resolved


def _log_changes(original: str, resolved: str):
    if original == resolved:
        return
    diff = abs(len(resolved.split()) - len(original.split()))
    print(f"[coref] ✓ {diff} token(s) changed ({len(original)} → {len(resolved)} chars)")
