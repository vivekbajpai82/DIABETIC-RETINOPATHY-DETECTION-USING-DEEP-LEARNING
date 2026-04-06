from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import gc  # Memory cleanup ke liye
from utils import predict, phase1_model, phase2_model, phase3_model
from utils import transform_phase1, transform_phase2, transform_phase3
from quality_check import check_image_quality, get_quality_recommendations
from gradcam_utils import generate_gradcam_heatmap, generate_multiphase_heatmaps

# -------------------------------
# App initialization
# -------------------------------
app = FastAPI(title="Diabetic Retinopathy Detection API with GradCAM")

# -------------------------------
# CORS Settings
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def root():
    return {"status": "Backend running 🚀", "features": ["prediction", "quality_check", "gradcam"]}

# -------------------------------
# Image Quality Check Endpoint
# -------------------------------
@app.post("/check_quality")
async def check_quality(file: UploadFile = File(...)):
    """Check image quality without running prediction"""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        quality_result = check_image_quality(image)
        recommendations = get_quality_recommendations(quality_result)
        
        return {
            "quality_check": quality_result,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quality check failed: {str(e)}"
        )
    finally:
        # Memory Cleanup
        if 'image_bytes' in locals(): del image_bytes
        if 'image' in locals(): del image
        gc.collect()

# -------------------------------
# Prediction Endpoint (with Quality Check)
# -------------------------------
@app.post("/predict")
async def predict_dr(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Quality Check
        quality_result = check_image_quality(image)
        
        if quality_result["status"] == "rejected":
            return {
                "success": False,
                "error": "Image quality insufficient",
                "quality_check": quality_result,
                "recommendations": get_quality_recommendations(quality_result)
            }
        
        # Run Prediction
        prediction_result = predict(image)
        
        return {
            "success": True,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "quality_check": quality_result,
            "recommendations": get_quality_recommendations(quality_result) if quality_result["status"] == "warning" else None
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Memory Cleanup
        if 'image_bytes' in locals(): del image_bytes
        if 'image' in locals(): del image
        gc.collect()

# -------------------------------
# GradCAM Heatmap Endpoint
# -------------------------------
@app.post("/predict_with_heatmap")
async def predict_with_heatmap(file: UploadFile = File(...)):
    """
    Prediction + GradCAM heatmaps for ALL phases
    Shows where each model is looking
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Quality Check
        quality_result = check_image_quality(image)
        
        if quality_result["status"] == "rejected":
            return {
                "success": False,
                "error": "Image quality insufficient",
                "quality_check": quality_result
            }
        
        # Run Prediction
        prediction_result = predict(image)
        
        # Generate GradCAM heatmaps for all phases
        heatmaps = generate_multiphase_heatmaps(
            phase1_model, phase2_model, phase3_model,
            image, transform_phase1, transform_phase2, transform_phase3
        )
        
        return {
            "success": True,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "quality_check": quality_result,
            "heatmaps": heatmaps
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction with heatmap failed: {str(e)}"
        )
    finally:
        # Memory Cleanup
        if 'image_bytes' in locals(): del image_bytes
        if 'image' in locals(): del image
        gc.collect()

# -------------------------------
# Single Phase GradCAM
# -------------------------------
@app.post("/gradcam/{phase}")
async def get_gradcam_for_phase(phase: int, file: UploadFile = File(...)):
    """
    Get GradCAM for specific phase only
    phase: 1, 2, or 3
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if phase == 1:
            x = transform_phase1(image).unsqueeze(0)
            heatmap = generate_gradcam_heatmap(
                phase1_model, x, image, model_type="efficientnet"
            )
        elif phase == 2:
            x = transform_phase2(image).unsqueeze(0)
            heatmap = generate_gradcam_heatmap(
                phase2_model, x, image, model_type="timm_efficientnet"
            )
        elif phase == 3:
            x = transform_phase3(image).unsqueeze(0)
            heatmap = generate_gradcam_heatmap(
                phase3_model, x, image, model_type="efficientnet"
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid phase. Must be 1, 2, or 3")
        
        return heatmap

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"GradCAM generation failed: {str(e)}"
        )
    finally:
        # Memory Cleanup
        if 'image_bytes' in locals(): del image_bytes
        if 'image' in locals(): del image
        if 'x' in locals(): del x
        gc.collect()

# -------------------------------
# Batch Quality Check
# -------------------------------
@app.post("/batch_quality_check")
async def batch_quality_check(files: list[UploadFile] = File(...)):
    """Check quality of multiple images at once"""
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            quality_result = check_image_quality(image)
            
            results.append({
                "filename": file.filename,
                "quality_check": quality_result
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            # Memory Cleanup for each image in batch
            if 'image_bytes' in locals(): del image_bytes
            if 'image' in locals(): del image
            gc.collect()
            
    return {"results": results}