from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from utils import predict

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "DR Detection Backend is Running (Hybrid Mode)"}

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Call the hybrid predict function from utils
        result = predict(image)
        return result
        
    except Exception as e:
        return {"error": str(e)}