# core/file_parser.py — Đọc và trích xuất text từ nhiều định dạng file
"""
Hỗ trợ:
  .txt            → _read_text_smart (giữ nguyên logic cũ, không cần Kreuzberg)
  .pdf            → Kreuzberg (giữ layout bảng số liệu, heading hierarchy)
  .docx / .doc    → Kreuzberg
  .xlsx / .xls    → Kreuzberg
  .pptx / .ppt    → Kreuzberg
  .html / .htm    → Kreuzberg
  .md             → Kreuzberg

Output luôn là str thuần (plain text) → đưa thẳng vào hierarchical_split().

Post-processing PDF:
  _strip_page_headers() loại bỏ các dòng page-header lặp lại do PDF inject
  (vd: "2 CTCP SỮA VIỆT NAM (VNM) - BẢN TIN NHÀ ĐẦU TƯ QUÝ 1 NĂM 2026")
  để nội dung chảy liên tục qua các trang, chunker mới cắt đúng ranh giới.
"""

import os
import re
from pathlib import Path

# ── Định dạng Kreuzberg xử lý ────────────────────────────────────────────────
KREUZBERG_EXTENSIONS = {
    ".pdf", ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".html", ".htm",
    ".md",
}

# ── Pattern page-header inject bởi PDF (lặp mỗi trang) ──────────────────────
# Dạng: "<số> CTCP SỮA VIỆT NAM..." hoặc các header tương tự
# Thêm pattern mới vào list nếu gặp loại tài liệu khác
_PDF_PAGE_HEADER_PATTERNS: list[re.Pattern] = [
    # "2 CTCP SỮA VIỆT NAM (VNM) - BẢN TIN NHÀ ĐẦU TƯ QUÝ 1 NĂM 2026"
    re.compile(
        r'^\s*\d+\s+CTCP\s+S[ỮƯ]A\s+VI[ỆE]T\s+NAM.*$',
        re.IGNORECASE,
    ),
    # Header dạng "Trang X / Y" hoặc "Page X of Y"
    re.compile(
        r'^\s*(?:Trang|Page)\s+\d+\s*(?:/|of)\s*\d+\s*$',
        re.IGNORECASE,
    ),
    # Header chỉ gồm số trang đứng một mình trên dòng
    re.compile(r'^\s*\d{1,4}\s*$'),
]


def _strip_pdf_page_headers(text: str) -> str:
    """
    Loại bỏ các dòng page-header lặp lại do PDF inject vào giữa nội dung.

    Nguyên tắc:
    - Scan từng dòng, nếu khớp bất kỳ pattern nào → bỏ dòng đó
    - Sau khi xóa dòng header, nối đoạn văn trước và sau để text chảy liền
      (thay thế blank line thừa bằng đúng 1 blank line)
    - Giữ nguyên tất cả dòng khác (kể cả dòng trắng cấu trúc đoạn văn)

    Kết quả: chunker nhận text liền mạch, không bị đứt đoạn tại page break.
    """
    lines = text.split('\n')
    cleaned: list[str] = []

    for line in lines:
        is_header = any(pat.match(line) for pat in _PDF_PAGE_HEADER_PATTERNS)
        if is_header:
            print(f"  [file_parser] strip page-header: {line.strip()!r}")
            # Không thêm dòng này — nội dung tiếp tục chảy liền
            continue
        cleaned.append(line)

    result = '\n'.join(cleaned)

    # Collapse chuỗi blank line > 2 thành đúng 2 (1 blank = phân cách đoạn)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


# ── Fallback: đọc .txt với auto-detect encoding ───────────────────────────────
def _read_text_smart(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            print(f"  Encoding detected: {enc}")
            return text
        except UnicodeDecodeError:
            pass
    try:
        import chardet
        result = chardet.detect(raw)
        detected   = result.get("encoding") or ""
        confidence = result.get("confidence", 0)
        print(f"  chardet: {detected} (confidence={confidence:.0%})")
        if detected and confidence > 0.7:
            try:
                text = raw.decode(detected)
                print(f"  Encoding detected: {detected}")
                return text
            except (UnicodeDecodeError, LookupError):
                pass
    except ImportError:
        print("  chardet chưa cài — thử các encoding phổ biến")
    for enc in ("cp1258", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            print(f"  Encoding detected (fallback): {enc}")
            return text
        except (UnicodeDecodeError, LookupError):
            pass
    print("  ⚠️  WARNING: fallback utf-8 errors=replace")
    return raw.decode("utf-8", errors="replace")


# ── Kreuzberg extractor ───────────────────────────────────────────────────────
async def _extract_with_kreuzberg(path: Path) -> str:
    """
    Dùng Kreuzberg để trích xuất text từ PDF / DOCX / XLSX / PPTX / HTML / MD.
    - output_format="plain"            → giữ căn chỉnh bảng số liệu
    - include_document_structure=True  → giữ heading hierarchy
    - KHÔNG dùng ChunkingConfig        → để chunker.py của dự án tự xử lý

    Với PDF: sau khi extract, gọi _strip_pdf_page_headers() để loại bỏ
    các dòng header lặp lại giữa các trang → text chảy liên tục.
    """
    try:
        import kreuzberg
    except ImportError:
        raise ImportError(
            "Kreuzberg chưa được cài. Chạy: pip install kreuzberg"
        )

    config = kreuzberg.ExtractionConfig(
        output_format="plain",
        include_document_structure=True,
    )

    print(f"  Kreuzberg extracting: {path.name} ...")
    result = await kreuzberg.extract_file(path, config=config)

    # Ưu tiên lấy text đầy đủ từ result.content
    text = getattr(result, "content", "") or ""

    # Nếu content rỗng → ghép từ chunks (fallback)
    if not text.strip():
        chunks = getattr(result, "chunks", []) or []
        parts  = []
        for chunk in chunks:
            content = getattr(chunk, "content", "") or ""
            if content.strip():
                parts.append(content.replace(os.linesep, "\n").replace("\r", "\n"))
        text = "\n\n".join(parts)

    if not text.strip():
        raise ValueError(f"Kreuzberg không trích xuất được text từ: {path.name}")

    # Chuẩn hóa line break
    text = text.replace(os.linesep, "\n").replace("\r", "\n")
    print(f"  Kreuzberg OK — {len(text):,} chars extracted")

    # ── Post-processing PDF: strip page headers ──────────────────────────────
    if path.suffix.lower() == ".pdf":
        text_before = len(text)
        text = _strip_pdf_page_headers(text)
        print(f"  Page-header strip: {text_before - len(text):+,} chars removed")

    return text


# ── Public API ────────────────────────────────────────────────────────────────
async def parse_file(path: Path) -> str:
    """
    Đọc file và trả về plain text, bất kể định dạng.

    Args:
        path: Path tới file cần đọc.

    Returns:
        str: Toàn bộ nội dung văn bản của file (page headers đã được strip với PDF).

    Raises:
        FileNotFoundError: File không tồn tại.
        ValueError:        Định dạng không hỗ trợ hoặc trích xuất thất bại.
    """
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {path.resolve()}")

    ext = path.suffix.lower()

    if ext == ".txt":
        return _read_text_smart(path)

    if ext in KREUZBERG_EXTENSIONS:
        return await _extract_with_kreuzberg(path)

    raise ValueError(
        f"Định dạng '{ext}' chưa được hỗ trợ.\n"
        f"Hỗ trợ: .txt, {', '.join(sorted(KREUZBERG_EXTENSIONS))}"
    )
