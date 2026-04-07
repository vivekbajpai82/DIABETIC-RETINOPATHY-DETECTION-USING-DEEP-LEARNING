import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import gc
from huggingface_hub import hf_hub_download

# ==========================================
# CONFIGURATION
# ==========================================
REPO_ID = "vivekbajpai82/dr-models"
MODEL_FILES = ["phase_1.pth", "phase_2.pth", "phase_3.pth"]
TOKEN = os.getenv("HF_TOKEN")

# ==========================================
# TRANSFORMS (JO MISSING THE ❌)
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
# SMART & LIGHTWEIGHT LOADER
# ==========================================
def smart_load(model, path):
    gc.collect() # Har load se pehle safai
    state_dict = torch.load(path, map_location="cpu")
    new_state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    del state_dict
    model.eval()
    return model

# ==========================================
# PREDICTION LOGIC (MEMORY OPTIMIZED)
# ==========================================
@torch.no_grad()
def predict(image: Image.Image):
    image = image.convert("RGB")
    
    try:
        # --- PHASE 1 ---
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model = smart_load(model, "models/phase_1.pth")
        
        img = transform_phase1(image).unsqueeze(0)
        output = model(img)
        p1_pred = torch.argmax(output, 1).item()
        p1_conf = torch.softmax(output, dim=1)[0][p1_pred].item()
        
        del model, output
        gc.collect()

        if p1_pred == 0:
            return {"prediction": "No_DR", "confidence": round(p1_conf * 100, 2)}

        # --- PHASE 2 ---
        model = models.efficientnet_b5(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        model = smart_load(model, "models/phase_2.pth")
        
        img = transform_phase23(image).unsqueeze(0)
        output = model(img)
        p2_pred = torch.argmax(output, 1).item()
        p2_conf = torch.softmax(output, dim=1)[0][p2_pred].item()
        
        del model, output
        gc.collect()

        if p2_pred == 0: return {"prediction": "Mild", "confidence": round(p2_conf * 100, 2)}
        if p2_pred == 1: return {"prediction": "Moderate", "confidence": round(p2_conf * 100, 2)}

        # --- PHASE 3 ---
        model = models.efficientnet_b5(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model = smart_load(model, "models/phase_3.pth")
        
        output = model(img)
        p3_pred = torch.argmax(output, 1).item()
        p3_conf = torch.softmax(output, dim=1)[0][p3_pred].item()
        
        labels = ["Severe", "Proliferative"]
        res = {"prediction": labels[p3_pred], "confidence": round(p3_conf * 100, 2)}
        
        del model, output, img
        gc.collect()
        return res

    except Exception as e:
        gc.collect()
        return {"error": str(e)}