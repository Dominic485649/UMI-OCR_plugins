"""Narrow PaddleX's VL dependency gate to the bundled OCR-core closure.

PaddleOCR-VL with the external llama.cpp server uses the PaddleX layout and
OCR-core runtime.  The upstream pipeline declares the much larger ``ocr``
extra, which would require unrelated document-parser packages and would make
the locked offline closure unnecessarily non-deterministic.  Keep this small,
auditable source patch separate from the upstream wheel.
"""

from __future__ import annotations

import sys
from pathlib import Path


OLD = '@pipeline_requires_extra("ocr")'
NEW = '@pipeline_requires_extra("ocr-core")'


def apply(site_packages: str) -> Path:
    root = Path(site_packages).resolve()
    path = root / "paddlex" / "inference" / "pipelines" / "paddleocr_vl" / "pipeline.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if NEW in text and OLD not in text:
        return path
    count = text.count(OLD)
    if count != 3:
        raise RuntimeError(f"Expected three PaddleOCR-VL dependency decorators, found {count}")
    path.write_text(text.replace(OLD, NEW), encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_paddlex_vl_3_7_2.py <site-packages>")
    print(apply(sys.argv[1]))
