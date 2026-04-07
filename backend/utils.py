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
    """Download models from Hugging Face if not present"""
    os.makedirs("models", exist_ok=True)

    for file in MODEL_FILES:
        dest_path = f"models/{file}"
        if not os.path.exists(dest_path):
            print(f"⬇ Downloading {file} from Hugging Face...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file,
                    local_dir="models",
                    token=TOKEN,
                    repo_type="dataset"
                )
                print(f"✅ {file} downloaded successfully")
            except Exception as e:
                print(f"❌ Error downloading {file}: {e}")

# Auto-download if token exists
if TOKEN:
    download_private_models()

# ==========================================
# TRANSFORMS
# ==========================================
transform_phase1 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_phase23 = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================
# MODEL LOADERS (🔥 SMART LOADER LAGA DIYA)
# ==========================================

def smart_load(model, path):
    """Ye function weights ke naam automatically theek karke load karega"""
    # 1. Weights file load karo
    state_dict = torch.load(path, map_location="cpu")
    
    # 2. 'module.' ya 'model.' jaisa kachra naam se hatao (Fix DataParallel keys)
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")  # GPU training prefix hataya
        k = k.replace("model.", "")   # Custom wrapper prefix hataya
        new_state_dict[k] = v
        
    # 3. Model mein weights dalo safely
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"⚠️ Warning: Exact match failed for {path}. Trying Safe Mode...")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()
    return model

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
# PREDICTION LOGIC
# ==========================================

@torch.no_grad()
def predict(image: Image.Image):
    try:
        # Ensure RGB
        image = image.convert("RGB")

        # =========================
        # Phase 1 → No_DR vs DR
        # =========================
        model = load_phase1()

        img = transform_phase1(image).unsqueeze(0)
        output = model(img)

        prob = torch.softmax(output, dim=1)
        conf, cls_idx = torch.max(prob, 1)

        phase1_pred = cls_idx.item()
        phase1_conf = conf.item()

        # Cleanup
        del model
        gc.collect()

        # If No DR → return directly
        if phase1_pred == 0:
            return {
                "prediction": "No_DR",
                "confidence": round(phase1_conf * 100, 2)
            }

        # =========================
        # Phase 2 → Mild / Moderate / Severe candidate
        # =========================
        model = load_phase2()

        img = transform_phase23(image).unsqueeze(0)
        output = model(img)

        prob = torch.softmax(output, dim=1)
        conf, cls_idx = torch.max(prob, 1)

        phase2_pred = cls_idx.item()
        phase2_conf = conf.item()

        del model
        gc.collect()

        if phase2_pred == 0:
            return {
                "prediction": "Mild",
                "confidence": round(phase2_conf * 100, 2)
            }

        if phase2_pred == 1:
            return {
                "prediction": "Moderate",
                "confidence": round(phase2_conf * 100, 2)
            }

        # =========================
        # Phase 3 → Severe vs Proliferative
        # =========================
        model = load_phase3()

        img = transform_phase23(image).unsqueeze(0)
        output = model(img)

        prob = torch.softmax(output, dim=1)
        conf, cls_idx = torch.max(prob, 1)

        labels = ["Severe", "Proliferative"]

        final_pred = labels[cls_idx.item()]
        final_conf = conf.item()

        del model
        gc.collect()

        return {
            "prediction": final_pred,
            "confidence": round(final_conf * 100, 2)
        }

    except Exception as e:
        print("❌ Prediction Error:", e)
        return {
            "prediction": None,
            "confidence": None,
            "error": str(e)
        }

    finally:
        gc.collect()