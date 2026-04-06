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
REPO_ID = "vivekbajpai82/dr-models"  # Tera Hugging Face Repo
MODEL_FILES = ["phase_1.pth", "phase_2.pth", "phase_3.pth"]
TOKEN = os.getenv("HF_TOKEN")

def download_private_models():
    """Hugging Face se models download karne ke liye"""
    os.makedirs("models", exist_ok=True)
    for file in MODEL_FILES:
        dest_path = f"models/{file}"
        if not os.path.exists(dest_path):
            print(f"Downloading {file} from Hugging Face...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file,
                    local_dir="models",
                    token=TOKEN,
                    repo_type="dataset"
                )
            except Exception as e:
                print(f"Error downloading {file}: {e}")


if TOKEN:
    download_private_models()

# ==========================================
# TRANSFORMS (Keep these global)
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
# RAM OPTIMIZED LOADERS (Load one by one)
# ==========================================

def load_phase1():
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("models/phase_1.pth", map_location='cpu'))
    model.eval()
    return model

def load_phase2():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load("models/phase_2.pth", map_location='cpu'))
    model.eval()
    return model

def load_phase3():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("models/phase_3.pth", map_location='cpu'))
    model.eval()
    return model

# ==========================================
# PREDICTION LOGIC (RAM Management)
# ==========================================

@torch.no_grad()
def predict(image: Image.Image):
    try:
        # Step 1: Phase 1 (No DR vs DR)
        model = load_phase1()
        img = transform_phase1(image).unsqueeze(0)
        output = model(img)
        prob = torch.softmax(output, dim=1)
        conf, cls_idx = torch.max(prob, 1)
        
        # RAM Khali karo
        del model
        gc.collect()

        if cls_idx.item() == 0:
            return {"prediction": "No_DR", "confidence": round(conf.item() * 100, 2)}

        # Step 2: Phase 2 (DR Severity Part 1)
        model = load_phase2()
        img = transform_phase23(image).unsqueeze(0)
        output = model(img)
        prob = torch.softmax(output, dim=1)
        conf, cls_idx = torch.max(prob, 1)

        del model
        gc.collect()

        if cls_idx.item() == 0:
            return {"prediction": "Mild", "confidence": round(conf.item() * 100, 2)}
        if cls_idx.item() == 1:
            return {"prediction": "Moderate", "confidence": round(conf.item() * 100, 2)}

        # Step 3: Phase 3 (DR Severity Part 2)
        model = load_phase3()
        img = transform_phase23(image).unsqueeze(0)
        output = model(img)
        prob = torch.softmax(output, dim=1)
        conf, cls_idx = torch.max(prob, 1)

        del model
        gc.collect()

        labels = ["Severe", "Proliferative"]
        return {"prediction": labels[cls_idx.item()], "confidence": round(conf.item() * 100, 2)}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Final safety cleanup
        gc.collect()