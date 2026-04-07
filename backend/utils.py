import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import gc
from huggingface_hub import hf_hub_download

# ==========================================
# CONFIGURATION & HUGGING FACE SETTINGS
# ==========================================
REPO_ID = "vivekbajpai82/dr-models"
MODEL_FILES = ["phase_1.pth", "phase_2.pth", "phase_3.pth"]
TOKEN = os.getenv("HF_TOKEN")

def download_private_models():
    """Hugging Face se models download karne ke liye"""
    os.makedirs("models", exist_ok=True)
    for file in MODEL_FILES:
        dest_path = f"models/{file}"
        if not os.path.exists(dest_path):
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file,
                    local_dir="models",
                    token=TOKEN,
                    repo_type="dataset"
                )
            except Exception as e:
                print(f"❌ Error downloading {file}: {e}")

# Auto-download startup par
if TOKEN:
    download_private_models()

# ==========================================
# TRANSFORMS (Main.py uses these)
# ==========================================
transform_phase1 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_phase23 = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# SMART LOADER (RAM Optimized)
# ==========================================
def smart_load(model, path):
    """Memory bachane ke liye weights load karne ka smart tareeka"""
    gc.collect() 
    if not os.path.exists(path):
        return model # Agar file nahi hai toh khali model dedo (startup fail nahi hoga)
    
    state_dict = torch.load(path, map_location="cpu")
    # Prefix hatana (module. ya model.)
    new_state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    del state_dict # Weights load hote hi memory khali karo
    model.eval()
    return model

# ==========================================
# MODEL LOADERS (Main.py imports these)
# ==========================================
def load_phase1():
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return smart_load(model, "models/phase_1.pth")

def load_phase2():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return smart_load(model, "models/phase_2.pth")

def load_phase3():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return smart_load(model, "models/phase_3.pth")

# ==========================================
# PREDICTION LOGIC (EXTREME MEMORY CLEANUP)
# ==========================================
@torch.no_grad()
def predict(image: Image.Image):
    """3-Phase prediction logic with aggressive memory management"""
    image = image.convert("RGB")
    
    try:
        # --- PHASE 1: No_DR vs DR ---
        model = load_phase1()
        img = transform_phase1(image).unsqueeze(0)
        output = model(img)
        
        p1_pred = torch.argmax(output, 1).item()
        p1_conf = torch.softmax(output, dim=1)[0][p1_pred].item()
        
        # Cleanup Phase 1
        del model, output
        gc.collect()

        if p1_pred == 0:
            return {"prediction": "No_DR", "confidence": round(p1_conf * 100, 2)}

        # --- PHASE 2: Mild / Moderate / Severe candidate ---
        model = load_phase2()
        img_large = transform_phase23(image).unsqueeze(0)
        output = model(img_large)
        
        p2_pred = torch.argmax(output, 1).item()
        p2_conf = torch.softmax(output, dim=1)[0][p2_pred].item()
        
        # Cleanup Phase 2
        del model, output
        gc.collect()

        if p2_pred == 0: 
            return {"prediction": "Mild", "confidence": round(p2_conf * 100, 2)}
        if p2_pred == 1: 
            return {"prediction": "Moderate", "confidence": round(p2_conf * 100, 2)}

        # --- PHASE 3: Severe vs Proliferative ---
        model = load_phase3()
        # Re-use img_large to save RAM
        output = model(img_large)
        
        p3_pred = torch.argmax(output, 1).item()
        p3_conf = torch.softmax(output, dim=1)[0][p3_pred].item()
        
        labels = ["Severe", "Proliferative"]
        res = {"prediction": labels[p3_pred], "confidence": round(p3_conf * 100, 2)}
        
        # Final Cleanup
        del model, output, img_large
        gc.collect()
        
        return res

    except Exception as e:
        gc.collect()
        return {"error": str(e)}