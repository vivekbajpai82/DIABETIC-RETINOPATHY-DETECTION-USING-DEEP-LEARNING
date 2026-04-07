import os
import gc
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

from utils import predict, transform_phase1, transform_phase23
from quality_check import check_image_quality
from utils import load_phase1, load_phase2, load_phase3
from gradcam_utils import generate_multiphase_heatmaps

# ==========================================
# APP INIT (🔥 MOST IMPORTANT)
# ==========================================
app = FastAPI(title="DR Detection API")

# ==========================================
# CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ROOT
# ==========================================
@app.get("/")
def root():
    return {"status": "Backend running 🚀"}

# ==========================================
# PREDICT
# ==========================================
@app.post("/predict")
async def predict_dr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Quality Check
        quality_result = check_image_quality(image)

        if quality_result["status"] == "rejected":
            return {
                "success": False,
                "error": "Low quality image",
                "quality_check": quality_result
            }

        # Prediction
        result = predict(image)
        print("DEBUG:", result)

        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "quality_check": quality_result
            }

        return {
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "quality_check": quality_result
        }

    except Exception as e:
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        gc.collect()

# ==========================================
# HEATMAP
# ==========================================
@app.post("/predict_with_heatmap")
async def predict_with_heatmap(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Load models
        m1 = load_phase1()
        m2 = load_phase2()
        m3 = load_phase3()

        result = predict(image)

        if result.get("error"):
            return {"success": False, "error": result["error"]}

        heatmaps = generate_multiphase_heatmaps(
            m1, m2, m3,
            image,
            transform_phase1,
            transform_phase23,
            transform_phase23
        )

        del m1, m2, m3
        gc.collect()

        return {
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "heatmaps": heatmaps
        }

    except Exception as e:
        print("❌ Heatmap Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        gc.collect()

# ==========================================
# QUALITY CHECK
# ==========================================
@app.post("/check_quality")
async def check_quality_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        quality_result = check_image_quality(image)

        return {
            "success": True,
            "quality_check": quality_result
        }

    except Exception as e:
        print("❌ Quality Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        gc.collect()

# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)