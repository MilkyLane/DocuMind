import pytesseract
import pdf2image
from PIL import Image
from pathlib import Path
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
PDF_EXT = ".pdf"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def ocr_image(image_path: Path) -> str:
    img = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(img).strip()

def ocr_pdf(pdf_path: Path) -> str:
    pages = pdf2image.convert_from_path(str(pdf_path))
    return "\n".join(pytesseract.image_to_string(page) for page in pages).strip()

def ocr_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in IMAGE_EXTS:
        return ocr_image(file_path)
    if ext == PDF_EXT:
        return ocr_pdf(file_path)
    raise ValueError(f"Unsupported file type: {ext}")

def run(target_class: str = None):
    dirs = (
        [RAW_DIR / target_class]
        if target_class
        else [d for d in RAW_DIR.iterdir() if d.is_dir()]
    )

    for class_dir in dirs:
        if not class_dir.is_dir():
            print(f"Directory not found: {class_dir}")
            continue

        out_class_dir = OUT_DIR / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        files = [
            f for f in class_dir.glob("*")
            if f.suffix.lower() in IMAGE_EXTS | {PDF_EXT}
        ]

        for file_path in tqdm(files, desc=f"OCR {class_dir.name}"):
            out_file = out_class_dir / f"{file_path.stem}.txt"
            try:
                text = ocr_file(file_path)
                out_file.write_text(text, encoding="utf-8")
            except Exception as e:
                print(f"Failed on {file_path}: {e}")

if __name__ == "__main__":
    run(target_class="resume")
