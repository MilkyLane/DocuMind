import pytesseract
from PIL import Image
from pathlib import Path
import pdf2image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
