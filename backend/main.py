from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
 import io
import gc
import uvicorn


from utils import predict, transform_phase1, transform_phase23
from quality_check import check_image_quality, get_quality_recommendations
# GradCAM ke liye models ko on-the-fly load karna hoga, isliye utils ke loaders chahiye
from utils import load_phase1, load_phase2, load_phase3
from gradcam_utils import generate_gradcam_heatmap, generate_multiphase_heatmaps

app = FastAPI(title="Diabetic Retinopathy Detection API - Optimized")

# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend running 🚀", "mode": "RAM Optimized"}

@app.post("/predict")
async def predict_dr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Quality Check
        quality_result = check_image_quality(image)
        if quality_result["status"] == "rejected":
            return {
                "success": False,
                "error": "Image quality insufficient",
                "quality_check": quality_result
            }
        
        # 2. Optimized Prediction (RAM safe)
        prediction_result = predict(image)
        
        return {
            "success": True,
            "prediction": prediction_result.get("prediction"),
            "confidence": prediction_result.get("confidence"),
            "quality_check": quality_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()

@app.post("/predict_with_heatmap")
async def predict_with_heatmap(file: UploadFile = File(...)):
    """Prediction + GradCAM (Slightly heavy on RAM)"""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Models ko on-demand load karo heatmaps ke liye
        m1, m2, m3 = load_phase1(), load_phase2(), load_phase3()
        
        prediction_result = predict(image)
        
        heatmaps = generate_multiphase_heatmaps(
            m1, m2, m3,
            image, transform_phase1, transform_phase23, transform_phase23
        )
        
        # Cleanup models immediately
        del m1, m2, m3
        gc.collect()

        return {
            "success": True,
            "prediction": prediction_result.get("prediction"),
            "heatmaps": heatmaps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)