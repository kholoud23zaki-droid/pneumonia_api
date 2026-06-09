import os
import threading
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import gdown
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "pneumonia_classifier.keras"
GDRIVE_FILE_ID = "1p0dewjOLBhgXcmJmv5eUV7x40J_ab2or"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

CLASS_LABELS = ["Lung_Opacity", "Normal", "Viral Pneumonia"]
IMG_SIZE = (224, 224)

# ── Model Container (thread-safe read/write) ──────────────────────────────────
_model = None
_model_lock = threading.Lock()


def get_model():
    with _model_lock:
        return _model


def set_model(m):
    global _model
    with _model_lock:
        _model = m


# ── Download + Load ───────────────────────────────────────────────────────────
def download_model():
    """Download model from Google Drive with retry and size validation."""
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"Model already exists. Size: {size_mb:.1f} MB")
        return

    print("Downloading model from Google Drive...")
    last_error = None

    for attempt in range(1, 3):
        # نظف أي فايل ناقص من محاولة سابقة
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        try:
            print(f"Download attempt {attempt}/2 ...")
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True, use_cookies=False)
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} raised exception: {e}")
            continue

        # تحقق من الحجم مباشرة بعد كل محاولة
        downloaded_size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        if downloaded_size >= 10 * 1024 * 1024:
            size_mb = downloaded_size / 1024 / 1024
            print(f"Model downloaded successfully. Size: {size_mb:.1f} MB")
            return  # نجح
        else:
            last_error = RuntimeError(
                f"Attempt {attempt}: file too small after download ({downloaded_size} bytes). "
                "Possible Google Drive quota/permissions issue."
            )
            print(last_error)

    # كل المحاولات فشلت — نظف وارفع exception
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    raise RuntimeError(f"Model download failed after 2 attempts. Last error: {last_error}")


def load_model():
    """Download and load model into memory (called once at startup)."""
    download_model()
    print("Loading model into memory...")
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    set_model(m)
    print("Model loaded and ready.")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        print(f"FATAL: Could not load model at startup: {e}")
        raise
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pneumonia Classifier API",
    description="EfficientNetB0 model — classifies lung X-rays into: Lung Opacity / Normal / Viral Pneumonia",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Matches notebook exactly:
    #   image.resize((224,224)) — no resampling arg → Pillow default BICUBIC
    #   img_to_array equivalent: np.array float32, values 0-255, NO /255 normalization
    #   ImageDataGenerator color_mode='rgb' → .convert("RGB") is correct
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)                      # default BICUBIC
    img_array = np.array(image, dtype=np.float32)      # (224, 224, 3), values 0-255
    img_array = np.expand_dims(img_array, axis=0)      # (1, 224, 224, 3)
    return img_array


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Pneumonia Classifier API is running",
        "classes": CLASS_LABELS,
        "input_shape": "224x224 RGB image",
        "endpoints": {"predict": "POST /predict", "health": "GET /health"},
    }


@app.get("/health")
def health():
    loaded = get_model() is not None
    return {
        "status": "healthy" if loaded else "loading",
        "model_loaded": loaded,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    m = get_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model is still loading — please retry in a moment")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpeg, png, etc.)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    try:
        predictions = m.predict(img_array, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = float(predictions[0][predicted_index])

    all_probs = {
        CLASS_LABELS[i]: f"{round(float(predictions[0][i]) * 100, 2)}%"
        for i in range(len(CLASS_LABELS))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": f"{round(confidence * 100, 2)}%",
        "all_probabilities": all_probs,
    }
