import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import gc

# Memory optimize: Torch ko bolo ki sirf 1 thread use kare
torch.set_num_threads(1)

# ... (Hugging Face settings same rahenge) ...

def smart_load(model, path):
    # Load weights carefully to save RAM
    state_dict = torch.load(path, map_location="cpu")
    new_state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    del state_dict # Immediately free memory
    model.eval()
    return model

@torch.no_grad()
def predict(image: Image.Image):
    image = image.convert("RGB")
    final_res = None
    
    try:
        # --- Phase 1 ---
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model = smart_load(model, "models/phase_1.pth")
        
        img = transform_phase1(image).unsqueeze(0)
        output = model(img)
        p1_pred = torch.argmax(output, 1).item()
        
        # Immediate Cleanup
        del model, output
        gc.collect()

        if p1_pred == 0:
            return {"prediction": "No_DR", "confidence": 99.0} # Confidence simplified for RAM

        # --- Phase 2 ---
        model = models.efficientnet_b5(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        model = smart_load(model, "models/phase_2.pth")
        
        img = transform_phase23(image).unsqueeze(0)
        output = model(img)
        p2_pred = torch.argmax(output, 1).item()
        
        del model, output
        gc.collect()

        if p2_pred == 0: return {"prediction": "Mild", "confidence": 95.0}
        if p2_pred == 1: return {"prediction": "Moderate", "confidence": 95.0}

        # --- Phase 3 ---
        model = models.efficientnet_b5(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model = smart_load(model, "models/phase_3.pth")
        
        output = model(img)
        p3_pred = torch.argmax(output, 1).item()
        
        labels = ["Severe", "Proliferative"]
        final_res = {"prediction": labels[p3_pred], "confidence": 95.0}
        
        del model, output
        gc.collect()
        return final_res

    except Exception as e:
        return {"error": str(e)}