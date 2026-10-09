"""从受控 DOCX 独立提取解析基线（§5.1 步骤1-2）。

不依赖 RAGFlow：直接读 word/document.xml，输出段落/标题/表格/图片/
嵌入对象/数字单位及其位置，供 parse_benchmark.py 做全量自动比对。
人工只复核异常与安全关键内容（§5.3）。

用法：
    python scripts/extract_docx_baseline.py --docx 600MW规程.docx \
        --out datasets/parse_baseline.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

NUMBER_UNIT_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*(?:MPa|kPa|Pa|kV|V|A|mA|℃|°C|N·m|Nm|r/min|mm|m\b|t/h|MW|kW|%)")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        import docx  # noqa: E402
    except ImportError:
        print("需安装 python-docx：uv add python-docx / pip install python-docx",
              file=sys.stderr)
        sys.exit(2)
    doc = docx.Document(args.docx)
    file_hash = sha256_of(args.docx)
    heading_stack: list[str] = []
    records: list[dict] = []

    def chapter_path() -> str:
        return " > ".join(heading_stack)

    for pi, p in enumerate(doc.paragraphs):
        text = (p.text or "").strip()
        style = (p.style.name if p.style else "")
        is_heading = style.startswith("Heading") or re.match(r"^\d+(\.\d+)*\s+\S", text or "")
        if is_heading and text:
            level = 1
            m = re.match(r"Heading (\d+)", style)
            if m:
                level = int(m.group(1))
            heading_stack = heading_stack[: max(0, level - 1)] + [text]
            records.append({"record_id": f"P{pi:05d}", "kind": "heading",
                            "text": text, "chapter_path": chapter_path(),
                            "position": {"para": pi}, "style": style})
            continue
        if not text:
            continue
        rec: dict = {"record_id": f"P{pi:05d}", "kind": "paragraph",
                     "text": text, "chapter_path": chapter_path(),
                     "position": {"para": pi}, "style": style}
        records.append(rec)
        for nu in NUMBER_UNIT_RE.findall(text):
            records.append({"record_id": f"P{pi:05d}#NU", "kind": "number_unit",
                            "text": nu.strip(), "chapter_path": chapter_path(),
                            "position": {"para": pi}})

    for ti, tb in enumerate(doc.tables):
        for ri, row in enumerate(tb.rows):
            for ci, cell in enumerate(row.cells):
                t = cell.text.strip()
                if not t:
                    continue
                records.append({
                    "record_id": f"T{ti:03d}R{ri:02d}C{ci:02d}", "kind": "table_cell",
                    "text": t, "chapter_path": chapter_path(),
                    "position": {"table": ti, "row": ri, "col": ci,
                                 "n_rows": len(tb.rows),
                                 "n_cols": len(row.cells)},
                    "table_id": f"T{ti:03d}"})
    n_images = len(doc.inline_shapes)
    for i, shp in enumerate(doc.inline_shapes):
        records.append({"record_id": f"IMG{i:03d}", "kind": "image",
                        "text": "", "chapter_path": chapter_path(),
                        "position": {"image": i},
                        "size": (getattr(shp, "width", None), getattr(shp, "height", None))})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"record_id": "_FILE_", "kind": "file",
                            "source_file": args.docx, "source_hash": file_hash,
                            "n_paragraphs": len(doc.paragraphs),
                            "n_tables": len(doc.tables),
                            "n_images": n_images}, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    kinds: dict[str, int] = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"基线 {args.out}：{len(records)} 条 {kinds}，sha256={file_hash[:12]}…")


if __name__ == "__main__":
    main()
