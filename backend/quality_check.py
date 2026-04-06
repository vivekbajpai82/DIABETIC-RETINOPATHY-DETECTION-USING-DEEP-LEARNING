import cv2
import numpy as np
from PIL import Image

def check_image_quality(image: Image.Image):
    """
    Comprehensive image quality check for retinal fundus images
    Returns: dict with status and detailed feedback
    """
    
    # Convert PIL to numpy array
    img_array = np.array(image.convert('RGB'))
    
    issues = []
    warnings = []
    score = 100  # Start with perfect score
    
    # ============================================
    # 1. RESOLUTION CHECK
    # ============================================
    height, width = img_array.shape[:2]
    min_resolution = 224  # Minimum acceptable
    recommended_resolution = 512
    
    if width < min_resolution or height < min_resolution:
        issues.append(f"Resolution too low ({width}x{height}). Minimum: {min_resolution}x{min_resolution}")
        score -= 40
    elif width < recommended_resolution or height < recommended_resolution:
        warnings.append(f"Low resolution ({width}x{height}). Recommended: {recommended_resolution}x{recommended_resolution}")
        score -= 10
    
    # ============================================
    # 2. BRIGHTNESS CHECK
    # ============================================
    # Convert to grayscale for brightness analysis
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 30:
        issues.append(f"Image too dark (brightness: {brightness:.1f}/255). Please use better lighting.")
        score -= 30
    elif brightness < 50:
        warnings.append(f"Image slightly dark (brightness: {brightness:.1f}/255)")
        score -= 10
    elif brightness > 230:
        issues.append(f"Image overexposed (brightness: {brightness:.1f}/255). Reduce lighting.")
        score -= 30
    elif brightness > 200:
        warnings.append(f"Image very bright (brightness: {brightness:.1f}/255)")
        score -= 10
    
    # ============================================
    # 3. BLUR DETECTION (Laplacian Variance)
    # ============================================
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 50:
        issues.append(f"Image too blurry (sharpness: {laplacian_var:.1f}). Hold camera steady.")
        score -= 35
    elif laplacian_var < 100:
        warnings.append(f"Image slightly blurry (sharpness: {laplacian_var:.1f})")
        score -= 15
    
    # ============================================
    # 4. CONTRAST CHECK
    # ============================================
    contrast = gray.std()
    
    if contrast < 20:
        issues.append(f"Very low contrast ({contrast:.1f}). Image lacks detail.")
        score -= 20
    elif contrast < 35:
        warnings.append(f"Low contrast ({contrast:.1f})")
        score -= 10
    
    # ============================================
    # 5. COLOR DISTRIBUTION (Retinal Image Check)
    # ============================================
    # Retinal images should have dominant red channel
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]
    
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
    
    # Typical retinal images have red > green > blue
    if red_mean < green_mean or red_mean < blue_mean:
        warnings.append("Color distribution unusual for retinal image. Ensure proper fundus camera settings.")
        score -= 15
    
    # ============================================
    # 6. EDGE DETECTION (Image Structure)
    # ============================================
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    if edge_density < 0.01:
        warnings.append("Very few image details detected. Check focus and lighting.")
        score -= 10
    
    # ============================================
    # 7. CIRCULAR FUNDUS CHECK (Advanced)
    # ============================================
    # Retinal images typically have circular field of view
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=min(width, height) // 2
    )
    
    if circles is None:
        warnings.append("No circular fundus region detected. May not be a proper retinal image.")
        score -= 15
    
    # ============================================
    # FINAL DECISION
    # ============================================
    
    # Ensure score doesn't go negative
    score = max(0, score)
    
    # Determine status
    if issues:
        status = "rejected"
        message = "Image quality insufficient for accurate analysis"
    elif warnings and score < 70:
        status = "warning"
        message = "Image quality acceptable but not optimal"
    else:
        status = "passed"
        message = "Image quality good for analysis"
    
    return {
        "status": status,  # "passed", "warning", "rejected"
        "quality_score": score,
        "message": message,
        "issues": issues,
        "warnings": warnings,
        "metrics": {
            "resolution": f"{width}x{height}",
            "brightness": round(brightness, 1),
            "sharpness": round(laplacian_var, 1),
            "contrast": round(contrast, 1)
        }
    }


def get_quality_recommendations(quality_result):
    """
    Generate actionable recommendations based on quality check
    """
    recommendations = []
    
    if quality_result["status"] == "rejected":
        recommendations.append("🚫 Cannot proceed with analysis. Please address the following:")
        recommendations.extend([f"  • {issue}" for issue in quality_result["issues"]])
        
        recommendations.append("\n📸 Tips for better image:")
        recommendations.append("  • Use proper fundus camera with adequate lighting")
        recommendations.append("  • Ensure patient's eye is properly dilated")
        recommendations.append("  • Hold camera steady to avoid blur")
        recommendations.append("  • Center the optic disc in the image")
        
    elif quality_result["status"] == "warning":
        recommendations.append("⚠️ Analysis will proceed but results may be less accurate:")
        recommendations.extend([f"  • {warning}" for warning in quality_result["warnings"]])
        
        recommendations.append("\n💡 For better results:")
        recommendations.append("  • Retake image with improved lighting/focus if possible")
        recommendations.append("  • Verify results with a medical professional")
    
    else:
        recommendations.append("✅ Image quality excellent for analysis")
    
    return recommendations