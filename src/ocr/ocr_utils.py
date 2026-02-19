import os
import pytesseract
from PIL import Image
from pathlib import Path
import pdf2image

# On Linux (Docker/Railway) tesseract is on PATH — no explicit path needed.
# On Windows dev machines it must be set manually via env var or falls back
# to the default install location.
_win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_tesseract_cmd = os.environ.get("TESSERACT_CMD", _win_default)
if Path(_tesseract_cmd).exists():
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
# else: trust that `tesseract` is on PATH (Linux/Docker)

def ocr_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    return pytesseract.image_to_string(img)

def ocr_pdf(path: Path) -> str:
    pages = pdf2image.convert_from_path(path)
    text = []
    for page in pages:
        text.append(pytesseract.image_to_string(page))
    return "\n".join(text)

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg", ".tiff"]:
        return ocr_image(path)

    if suffix == ".pdf":
        return ocr_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix}")
