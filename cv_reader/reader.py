"""
cv_reader/reader.py
-------------------
Reads PDF, DOCX, and TXT CV files and returns clean plain text.
"""

import os
from pathlib import Path


def read_cv(file_path: str) -> str:
    """
    Extract plain text from a CV file.
    Supports: PDF, DOCX, TXT, MD
    """
    ext = Path(file_path).suffix.lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CV file not found: {file_path}")

    if ext == ".pdf":
        return _read_pdf(file_path)
    elif ext == ".docx":
        return _read_docx(file_path)
    elif ext in (".txt", ".md"):
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported CV format: {ext}. Use PDF, DOCX, or TXT.")


def _read_pdf(path: str) -> str:
    # Primary: pdfplumber (best quality)
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text
    except ImportError:
        pass

    # Fallback: PyPDF2
    try:
        import PyPDF2
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except ImportError:
        raise SystemExit("Install pdfplumber:  pip install pdfplumber")


def _read_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise SystemExit("Install python-docx:  pip install python-docx")
