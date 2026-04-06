import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
import timm
from torchvision import transforms
from PIL import Image
import gc
from huggingface_hub import hf_hub_download  # 🔥 Secure download library

# Explicitly set device to CPU
device = torch.device("cpu")

# ============================================================
# SECURE MODEL DOWNLOAD LOGIC (Hugging Face)
# ============================================================
def download_private_models():
    # Yeh token hum Render ke Dashboard me set karenge
    token = os.getenv("HF_TOKEN")
    repo_id = "vivekbajpai82/dr-models" # Tera HF username/repo
    
    model_files = ["phase_1.pth", "phase_2.pth", "phase_3.pth"]
    os.makedirs("models", exist_ok=True)

    for file in model_files:
        dest_path = f"models/{file}"
        if not os.path.exists(dest_path):
            print(f"Downloading {file} from Hugging Face...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir="models",
                    token=token,
                    repo_type="dataset"
                )
            except Exception as e:
                print(f"Error downloading {file}: {e}")

# Agar Render par deploy ho raha hai (HF_TOKEN set hai), toh download shuru karo
if os.getenv("HF_TOKEN"):
    download_private_models()

# ============================================================
# SEPARATE TRANSFORMS FOR EACH PHASE
# ============================================================

# Phase 1: Trained on 300x300
transform_phase1 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Phase 2: Trained on 512x512
transform_phase2 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Phase 3: Trained on 224x224
transform_phase3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# LOAD FUNCTIONS
# ============================================================
def load_phase1():
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load("models/phase_1.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_phase2():
    model = timm.create_model(
        "tf_efficientnet_b5_ns",
        pretrained=False,
        num_classes=3,
        drop_rate=0.4
    )
    state = torch.load("models/phase_2.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_phase3():
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )
    checkpoint = torch.load("models/phase_3.pth", map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# ============================================================
# INITIALIZATION
# ============================================================
# Note: Render par build ke waqt error na aaye isliye hum check kar rahe hain
if os.path.exists("models/phase_1.pth"):
    print("Loading Phase 1 model...")
    phase1_model = load_phase1()

    print("Loading Phase 2 model...")
    phase2_model = load_phase2()

    print("Loading Phase 3 model...")
    phase3_model = load_phase3()

    print("✅ All models loaded successfully!")
else:
    print("⚠️ Models not found locally. Waiting for secure download...")

# ============================================================
# PREDICTION PIPELINE
# ============================================================
@torch.no_grad()
def predict(image: Image.Image):
    try:
        # PHASE 1
        x1 = transform_phase1(image).unsqueeze(0)
        out1 = phase1_model(x1)
        prob1 = torch.softmax(out1, dim=1)
        conf1, cls1 = torch.max(prob1, 1)

        if cls1.item() == 0:
            return {"prediction": "No_DR", "confidence": round(conf1.item() * 100, 2)}

        # PHASE 2
        x2 = transform_phase2(image).unsqueeze(0)
        out2 = phase2_model(x2)
        prob2 = torch.softmax(out2, dim=1)
        conf2, cls2 = torch.max(prob2, 1)

        if cls2.item() == 0:
            return {"prediction": "Mild", "confidence": round(conf2.item() * 100, 2)}
        if cls2.item() == 1:
            return {"prediction": "Moderate", "confidence": round(conf2.item() * 100, 2)}

        # PHASE 3
        x3 = transform_phase3(image).unsqueeze(0)
        out3 = phase3_model(x3)
        prob3 = torch.softmax(out3, dim=1)
        conf3, cls3 = torch.max(prob3, 1)

        return {
            "prediction": ["Severe", "Proliferative"][cls3.item()],
            "confidence": round(conf3.item() * 100, 2)
        }
    
    finally:
        # 🔥 Force memory cleanup after every prediction
        gc.collect()