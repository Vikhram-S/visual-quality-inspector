import cv2
import numpy as np
from PIL import Image
import io
import math
import base64

def estimate_noise_fast(gray):
    """
    Immerkaer fast noise variance estimation method.
    Estimates the noise standard deviation from a grayscale image using a 3x3 Laplacian operator.
    """
    H, W = gray.shape
    M = np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]], dtype=np.float32)
    sigma = np.sum(np.abs(cv2.filter2D(gray.astype(np.float32), -1, M)))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6.0 * (W - 2) * (H - 2))
    return float(sigma)

def calculate_blockiness_index(gray):
    """
    Measures 8x8 grid block edge discontinuity (JPEG compression artifacts).
    High values indicate low JPEG quality / blocky corruption.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    
    gray_f = gray.astype(np.float32)
    v_diff = np.abs(gray_f[:, 7:-1:8] - gray_f[:, 8::8])
    v_mean_block_diff = np.mean(v_diff)
    
    v_non_diff = np.abs(gray_f[:, 3:-1:8] - gray_f[:, 4::8])
    v_mean_non_diff = np.mean(v_non_diff) + 1e-5
    
    h_diff = np.abs(gray_f[7:-1:8, :] - gray_f[8::8, :])
    h_mean_block_diff = np.mean(h_diff)
    
    h_non_diff = np.abs(gray_f[3:-1:8, :] - gray_f[4::8, :])
    h_mean_non_diff = np.mean(h_non_diff) + 1e-5
    
    blockiness = (v_mean_block_diff / v_mean_non_diff + h_mean_block_diff / h_mean_non_diff) / 2.0
    return float(blockiness)

def calculate_entropy(gray):
    """Calculates Shannon entropy of image intensity distribution."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.log2(hist + 1e-12)
    entropy = -np.sum(hist * logs)
    return float(entropy)

def extract_features_from_array(img_bgr):
    """
    Extracts numerical quality features from an OpenCV BGR numpy image array.
    Returns a dictionary of feature names -> float values.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "laplacian_var": 0.0,
            "tenengrad_val": 0.0,
            "fft_blur_ratio": 0.0,
            "mean_luminance": 0.0,
            "median_luminance": 0.0,
            "shadow_clip_pct": 100.0,
            "highlight_clip_pct": 0.0,
            "std_luminance": 0.0,
            "rms_contrast": 0.0,
            "noise_variance": 999.0,
            "laplacian_noise_est": 999.0,
            "mean_saturation": 0.0,
            "std_saturation": 0.0,
            "blockiness_index": 10.0,
            "entropy": 0.0,
            "is_valid_format": 0.0
        }

    features = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    features["is_valid_format"] = 1.0

    # 1. SHARPNESS
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features["laplacian_var"] = float(laplacian.var())

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(sobelx**2 + sobely**2)
    features["tenengrad_val"] = float(tenengrad)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    radius = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 > radius**2
    high_freq_power = float(np.mean(magnitude_spectrum[mask]))
    total_power = float(np.mean(magnitude_spectrum)) + 1e-6
    features["fft_blur_ratio"] = high_freq_power / total_power

    # 2. BRIGHTNESS / EXPOSURE
    v_channel = hsv[:, :, 2]
    features["mean_luminance"] = float(np.mean(v_channel))
    features["median_luminance"] = float(np.median(v_channel))
    
    total_pixels = float(v_channel.size)
    shadow_pixels = np.sum(v_channel < 15)
    highlight_pixels = np.sum(v_channel > 240)
    features["shadow_clip_pct"] = float((shadow_pixels / total_pixels) * 100.0)
    features["highlight_clip_pct"] = float((highlight_pixels / total_pixels) * 100.0)

    # 3. CONTRAST
    features["std_luminance"] = float(np.std(v_channel))
    norm_luminance = v_channel / 255.0
    features["rms_contrast"] = float(np.std(norm_luminance))

    # 4. NOISE
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - smoothed.astype(np.float32)
    features["noise_variance"] = float(np.var(residual))
    features["laplacian_noise_est"] = estimate_noise_fast(gray)

    # 5. SATURATION
    s_channel = hsv[:, :, 1]
    features["mean_saturation"] = float(np.mean(s_channel))
    features["std_saturation"] = float(np.std(s_channel))

    # 6. CORRUPTION / BLOCKINESS & ENTROPY
    features["blockiness_index"] = calculate_blockiness_index(gray)
    features["entropy"] = calculate_entropy(gray)

    return features

def extract_features_from_bytes(image_bytes: bytes):
    """Decodes image bytes and extracts quality features."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            res = extract_features_from_array(None)
            res["is_valid_format"] = 0.0
            return res
        return extract_features_from_array(img_bgr)
    except Exception:
        res = extract_features_from_array(None)
        res["is_valid_format"] = 0.0
        return res

def extract_features_from_file(file_path: str):
    """Loads an image file from disk and extracts features."""
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        res = extract_features_from_array(None)
        res["is_valid_format"] = 0.0
        return res
    return extract_features_from_array(img_bgr)

def generate_defect_heatmap(image_bytes: bytes, grid_cols=8, grid_rows=8):
    """
    Computes local sharpness/exposure/noise heatmaps across an 8x8 grid.
    Returns:
      1. Base64-encoded PNG image with semi-transparent color overlay (Red=Defect, Green=Pristine)
      2. Grid array of patch scores for UI rendering.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, None

        h, w, _ = img_bgr.shape
        cell_h = max(1, h // grid_rows)
        cell_w = max(1, w // grid_cols)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        overlay = img_bgr.copy()
        
        grid_data = []

        for r in range(grid_rows):
            row_cells = []
            for c in range(grid_cols):
                y1, y2 = r * cell_h, min(h, (r + 1) * cell_h)
                x1, x2 = c * cell_w, min(w, (c + 1) * cell_w)
                
                patch = gray[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                lap_var = float(cv2.Laplacian(patch, cv2.CV_64F).var())
                mean_lum = float(np.mean(patch))
                
                sharp_score = min(100.0, (lap_var / 120.0) * 100.0)
                exp_penalty = max(0.0, abs(mean_lum - 128.0) - 40.0)
                patch_score = max(0.0, min(100.0, sharp_score - exp_penalty))

                row_cells.append({
                    "row": r, "col": c,
                    "score": round(patch_score, 1),
                    "laplacian_var": round(lap_var, 1),
                    "mean_luminance": round(mean_lum, 1)
                })

                if patch_score < 45.0:
                    color = (0, 0, 220)    # Red
                elif patch_score < 70.0:
                    color = (0, 215, 255)  # Yellow
                else:
                    color = (0, 200, 0)    # Green

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), 1)

            grid_data.append(row_cells)

        blended = cv2.addWeighted(overlay, 0.40, img_bgr, 0.60, 0)
        _, buffer = cv2.imencode('.png', blended)
        heatmap_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

        return heatmap_base64, grid_data
    except Exception:
        return None, None

FEATURE_NAMES = [
    "laplacian_var", "tenengrad_val", "fft_blur_ratio",
    "mean_luminance", "median_luminance", "shadow_clip_pct", "highlight_clip_pct",
    "std_luminance", "rms_contrast", "noise_variance", "laplacian_noise_est",
    "mean_saturation", "std_saturation", "blockiness_index", "entropy", "is_valid_format"
]
