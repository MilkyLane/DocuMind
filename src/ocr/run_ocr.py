import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import pytesseract
from PIL import Image
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def ocr_image(image_path: Path) -> str:
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text.strip()

def run():
    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        out_class_dir = OUT_DIR / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*"))

        for img_path in tqdm(images, desc=f"OCR {class_dir.name}"):
            try:
                text = ocr_image(img_path)
                out_file = out_class_dir / f"{img_path.stem}.txt"

                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)

            except Exception as e:
                print(f"Failed on {img_path}: {e}")

if __name__ == "__main__":
    run()
