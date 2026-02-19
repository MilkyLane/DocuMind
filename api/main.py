from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import tempfile

from src.inference.predict import predict

app = FastAPI(
    title="DocuMind",
    description="Document classification and field extraction API",
    version="1.0.0"
)

@app.get("/")
def health():
    return {"status": "ok", "message": "DocuMind API is running"}

@app.post("/predict")
async def predict_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename

        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = predict(str(tmp_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return result
