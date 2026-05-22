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
"""

import asyncio
import os
from pathlib import Path

# ── Định dạng Kreuzberg xử lý ────────────────────────────────────────────────
KREUZBERG_EXTENSIONS = {
    ".pdf", ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".html", ".htm",
    ".md",
}


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
    - include_document_structure=True  → giữ heading hierarchy (heading sẽ nằm trên dòng riêng)
    - KHÔNG dùng ChunkingConfig        → để chunker.py của dự án tự xử lý
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
    return text


# ── Public API ────────────────────────────────────────────────────────────────
async def parse_file(path: Path) -> str:
    """
    Đọc file và trả về plain text, bất kể định dạng.

    Args:
        path: Path tới file cần đọc.

    Returns:
        str: Toàn bộ nội dung văn bản của file.

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
