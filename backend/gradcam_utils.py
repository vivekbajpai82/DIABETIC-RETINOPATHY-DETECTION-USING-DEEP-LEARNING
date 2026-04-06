import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_target_layer(model, model_type):
    """
    Get appropriate target layer for GradCAM based on model architecture
    """
    if model_type == "efficientnet":
        # For torchvision EfficientNet (Phase 1 & 3)
        # Use last convolutional layer before classifier
        return [model.features[-1]]
    
    elif model_type == "timm_efficientnet":
        # For timm EfficientNet models
        # Try different layer names based on timm version
        if hasattr(model, 'conv_head'):
            return [model.conv_head]
        elif hasattr(model, 'blocks'):
            return [model.blocks[-1]]
        elif hasattr(model, 'features'):
            return [model.features[-1]]
        else:
            # Fallback: get last convolutional layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return [module]
    
    raise ValueError(f"Could not find appropriate target layer for model_type: {model_type}")


def generate_gradcam_heatmap(model, image_tensor, original_image, target_class=None, model_type="efficientnet"):
    """
    Generate GradCAM heatmap for the prediction
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        original_image: Original PIL Image
        target_class: Class to generate CAM for (None = predicted class)
        model_type: "efficientnet" or "timm_efficientnet"
    
    Returns:
        dict with base64 encoded heatmap and metadata
    """
    
    try:
        # Set model to eval mode
        model.eval()
        
        # ============================================
        # 1. SELECT TARGET LAYER
        # ============================================
        target_layers = get_target_layer(model, model_type)
        
        # ============================================
        # 2. GET PREDICTION IF TARGET NOT SPECIFIED
        # ============================================
        if target_class is None:
            with torch.no_grad():
                output = model(image_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # ============================================
        # 3. INITIALIZE GRADCAM WITH ERROR HANDLING
        # ============================================
        cam = GradCAM(
            model=model, 
            target_layers=target_layers
        )
        
        # ============================================
        # 4. GENERATE CAM
        # ============================================
        targets = [ClassifierOutputTarget(target_class)]
        
        # Requires grad for backprop
        image_tensor.requires_grad = True
        
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first image in batch
        
        # Clean up
        cam.activations_and_grads.release()
        
        # ============================================
        # 5. PREPARE ORIGINAL IMAGE FOR OVERLAY
        # ============================================
        # Resize original image to match heatmap size
        original_np = np.array(original_image.resize((grayscale_cam.shape[1], grayscale_cam.shape[0])))
        
        # Normalize to 0-1 range
        if original_np.max() > 1:
            original_np = original_np.astype(np.float32) / 255.0
        
        # ============================================
        # 6. CREATE OVERLAY
        # ============================================
        visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
        
        # ============================================
        # 7. CONVERT TO BASE64
        # ============================================
        # Convert numpy array to PIL Image
        heatmap_image = Image.fromarray(visualization)
        
        # Resize to reasonable size for frontend
        heatmap_image = heatmap_image.resize((512, 512), Image.LANCZOS)
        
        # Convert to base64
        buffered = io.BytesIO()
        heatmap_image.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # ============================================
        # 8. ALSO CREATE SIDE-BY-SIDE COMPARISON
        # ============================================
        original_resized = original_image.resize((512, 512), Image.LANCZOS)
        
        # Create side-by-side image
        comparison = Image.new('RGB', (1024, 512))
        comparison.paste(original_resized, (0, 0))
        comparison.paste(heatmap_image, (512, 0))
        
        # Add labels with PIL
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        
        try:
            # Try to use a nice font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Draw text with stroke for better visibility
        draw.text((20, 20), "Original", fill='white', font=font, stroke_width=2, stroke_fill='black')
        draw.text((532, 20), "Model Attention", fill='white', font=font, stroke_width=2, stroke_fill='black')
        
        # Convert comparison to base64
        buffered_comparison = io.BytesIO()
        comparison.save(buffered_comparison, format="PNG")
        comparison_base64 = base64.b64encode(buffered_comparison.getvalue()).decode('utf-8')
        
        # ============================================
        # 9. CALCULATE ATTENTION STATISTICS
        # ============================================
        attention_mean = float(np.mean(grayscale_cam))
        attention_max = float(np.max(grayscale_cam))
        attention_std = float(np.std(grayscale_cam))
        
        # Find regions with high attention (>70% of max)
        high_attention_mask = grayscale_cam > (0.7 * attention_max)
        high_attention_percentage = float(np.sum(high_attention_mask) / grayscale_cam.size * 100)
        
        return {
            "success": True,
            "heatmap_base64": f"data:image/png;base64,{heatmap_base64}",
            "comparison_base64": f"data:image/png;base64,{comparison_base64}",
            "target_class": int(target_class),
            "statistics": {
                "mean_attention": round(attention_mean, 3),
                "max_attention": round(attention_max, 3),
                "attention_std": round(attention_std, 3),
                "high_attention_area": round(high_attention_percentage, 2)
            }
        }
    
    except Exception as e:
        import traceback
        print(f"GradCAM Error: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "success": False,
            "error": f"Heatmap generation failed: {str(e)}"
        }


def generate_multiphase_heatmaps(phase1_model, phase2_model, phase3_model, 
                                  image, transform_phase1, transform_phase2, transform_phase3):
    """
    Generate heatmaps for all relevant phases in the prediction pipeline
    
    Returns:
        dict with heatmaps for each phase that was executed
    """
    
    results = {
        "phase1": None,
        "phase2": None,
        "phase3": None
    }
    
    try:
        # ============================================
        # PHASE 1: No DR vs Has DR
        # ============================================
        x1 = transform_phase1(image).unsqueeze(0)
        
        with torch.no_grad():
            out1 = phase1_model(x1)
            cls1 = torch.argmax(out1, dim=1).item()
        
        print(f"Phase 1 prediction: {cls1}")
        
        # Generate Phase 1 heatmap
        results["phase1"] = generate_gradcam_heatmap(
            phase1_model, x1, image, target_class=cls1, model_type="efficientnet"
        )
        
        if cls1 == 0:  # No DR - stop here
            print("No DR detected, stopping at Phase 1")
            return results
        
        # ============================================
        # PHASE 2: Mild/Moderate/Severe+
        # ============================================
        x2 = transform_phase2(image).unsqueeze(0)
        
        with torch.no_grad():
            out2 = phase2_model(x2)
            cls2 = torch.argmax(out2, dim=1).item()
        
        print(f"Phase 2 prediction: {cls2}")
        
        # Generate Phase 2 heatmap
        results["phase2"] = generate_gradcam_heatmap(
            phase2_model, x2, image, target_class=cls2, model_type="timm_efficientnet"
        )
        
        if cls2 in [0, 1]:  # Mild or Moderate - stop here
            print(f"Severity level {cls2}, stopping at Phase 2")
            return results
        
        # ============================================
        # PHASE 3: Severe vs Proliferative
        # ============================================
        x3 = transform_phase3(image).unsqueeze(0)
        
        with torch.no_grad():
            out3 = phase3_model(x3)
            cls3 = torch.argmax(out3, dim=1).item()
        
        print(f"Phase 3 prediction: {cls3}")
        
        # Generate Phase 3 heatmap
        results["phase3"] = generate_gradcam_heatmap(
            phase3_model, x3, image, target_class=cls3, model_type="efficientnet"
        )
        
        return results
    
    except Exception as e:
        import traceback
        print(f"Multiphase heatmap error: {str(e)}")
        print(traceback.format_exc())
        
        # Return whatever we have
        return results