import os
import io
import math
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_images"))

def create_synthetic_clean_image(width=512, height=512, pattern_idx=0):
    """
    Generates diverse, realistic synthetic clean images with rich textures, gradients, 
    geometric structures, and sharp details to serve as high-quality base images.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    pattern_type = pattern_idx % 6

    if pattern_type == 0:
        # Gradient background + geometric shapes + sharp text
        for y in range(height):
            r = int(120 + 130 * (y / height))
            g = int(80 + 100 * math.sin(y / 50.0))
            b = int(200 - 150 * (y / height))
            img[y, :] = [clamp(b), clamp(g), clamp(r)]
        
        cv2.circle(img, (width // 3, height // 3), 90, (255, 200, 50), -1)
        cv2.rectangle(img, (width // 2, height // 2), (width - 40, height - 40), (50, 240, 100), -1)
        cv2.line(img, (20, height - 30), (width - 20, 30), (255, 255, 255), 4)

    elif pattern_type == 1:
        # High contrast checkerboard & radial pattern
        grid_size = 32
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                if ((x // grid_size) + (y // grid_size)) % 2 == 0:
                    img[y:y+grid_size, x:x+grid_size] = [230, 230, 230]
                else:
                    img[y:y+grid_size, x:x+grid_size] = [40, 40, 50]
        cv2.circle(img, (width // 2, height // 2), 120, (30, 140, 255), 8)

    elif pattern_type == 2:
        # Fine texture grid with sharp lines
        X, Y = np.meshgrid(np.linspace(0, 10*np.pi, width), np.linspace(0, 10*np.pi, height))
        Z = np.sin(X) * np.cos(Y)
        norm_z = ((Z + 1) * 127.5).astype(np.uint8)
        img[:, :, 0] = norm_z
        img[:, :, 1] = np.rot90(norm_z)
        img[:, :, 2] = 255 - norm_z

    elif pattern_type == 3:
        # Multi-color scenery mockup (sky, sun, mountains, water)
        # Sky
        img[:int(height*0.5), :] = [235, 180, 100]  # Light orange sky
        # Sun
        cv2.circle(img, (int(width*0.7), int(height*0.25)), 50, (100, 255, 255), -1)
        # Mountain peaks
        pts1 = np.array([[0, int(height*0.6)], [int(width*0.35), int(height*0.25)], [int(width*0.6), int(height*0.6)]], np.int32)
        pts2 = np.array([[int(width*0.3), int(height*0.6)], [int(width*0.7), int(height*0.3)], [width, int(height*0.6)]], np.int32)
        cv2.fillPoly(img, [pts1], (80, 60, 40))
        cv2.fillPoly(img, [pts2], (60, 40, 30))
        # Ground/Water
        img[int(height*0.6):, :] = [180, 100, 40]

    elif pattern_type == 4:
        # High resolution document / architectural line drawing simulation
        img.fill(245)
        for i in range(10, width, 40):
            cv2.line(img, (i, 0), (i, height), (200, 200, 200), 1)
            cv2.line(img, (0, i), (width, i), (200, 200, 200), 1)
        cv2.rectangle(img, (60, 60), (width-60, height-60), (20, 20, 20), 3)
        cv2.putText(img, "TEST TARGET SHARPNESS 100%", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 180), 2)
        cv2.putText(img, "QUALITY CONTROL MATRIX", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 10, 10), 2)

    else:
        # Colorful spheres and noise-free rich details
        img.fill(30)
        colors = [(255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0), (238, 130, 238)]
        for i in range(8):
            cx = random.randint(50, width - 50)
            cy = random.randint(50, height - 50)
            r = random.randint(20, 70)
            color = colors[i % len(colors)]
            cv2.circle(img, (cx, cy), r, color, -1)
            cv2.circle(img, (cx, cy), r, (255, 255, 255), 2)

    return img

def clamp(val):
    return max(0, min(255, val))

# Degradation Generators
def apply_blur(img, severity=1):
    """Applies Gaussian / Motion Blur"""
    kernel_sizes = [7, 15, 25, 35]
    k_size = kernel_sizes[min(severity, len(kernel_sizes)-1)]
    return cv2.GaussianBlur(img, (k_size, k_size), 0)

def apply_underexposure(img, severity=1):
    """Reduces luminance/exposure (darkens)"""
    factors = [0.45, 0.30, 0.18, 0.08]
    factor = factors[min(severity, len(factors)-1)]
    inv_gamma = factor
    table = np.array([((i / 255.0) ** (1.0 / inv_gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_overexposure(img, severity=1):
    """Increases luminance/exposure (clips highlights)"""
    factors = [1.5, 2.0, 2.8, 3.8]
    factor = factors[min(severity, len(factors)-1)]
    res = img.astype(np.float32) * factor
    return np.clip(res, 0, 255).astype(np.uint8)

def apply_noise(img, severity=1):
    """Adds Gaussian or Salt-and-pepper noise"""
    sigmas = [20, 35, 55, 80]
    sigma = sigmas[min(severity, len(sigmas)-1)]
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_corruption(img, severity=1):
    """Applies heavy compression block artifacts and severe color/block degradation"""
    qualities = [12, 6, 3, 1]
    q = qualities[min(severity, len(qualities)-1)]
    
    # Compress via JPEG with low quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, result = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(result, cv2.IMREAD_COLOR)
    
    if severity >= 2:
        # Add synthetic blocking artifacts or missing scanlines
        h, w, _ = decoded.shape
        strip_y = random.randint(0, h - 40)
        decoded[strip_y:strip_y+30, :, :] = random.randint(0, 255)
    
    return decoded

def generate_dataset(num_clean_base=120, train_ratio=0.8):
    """
    Generates training and testing datasets with synthetic degradations.
    Total samples generated: num_clean_base * (1 clean + 5 defects * 3 severities) = 16 * num_clean_base images.
    For 120 base images = ~1920 labeled dataset items.
    Runs fast on CPU (< 10 seconds).
    """
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print(f"Generating synthetic dataset from {num_clean_base} base images...")

    categories = ["clean", "blur", "underexposed", "overexposed", "noise", "corrupted"]
    for cat in categories:
        os.makedirs(os.path.join(TRAIN_DIR, cat), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, cat), exist_ok=True)

    sample_saved = {cat: False for cat in categories}

    item_id = 0
    for i in range(num_clean_base):
        base_img = create_synthetic_clean_image(512, 512, pattern_idx=i)
        
        # Decide train or test split based on base_img index
        is_train = (i < int(num_clean_base * train_ratio))
        split_dir = TRAIN_DIR if is_train else TEST_DIR

        # 1. Clean version
        clean_path = os.path.join(split_dir, "clean", f"img_{i:04d}_clean.jpg")
        cv2.imwrite(clean_path, base_img)
        if not sample_saved["clean"]:
            cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_clean.jpg"), base_img)
            sample_saved["clean"] = True

        # 2. Defective versions with varying severities (0=mild, 1=moderate, 2=severe)
        for severity in range(3):
            # Blur
            img_blur = apply_blur(base_img, severity)
            cv2.imwrite(os.path.join(split_dir, "blur", f"img_{i:04d}_blur_s{severity}.jpg"), img_blur)
            if not sample_saved["blur"]:
                cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_blur.jpg"), img_blur)
                sample_saved["blur"] = True

            # Underexposed
            img_under = apply_underexposure(base_img, severity)
            cv2.imwrite(os.path.join(split_dir, "underexposed", f"img_{i:04d}_under_s{severity}.jpg"), img_under)
            if not sample_saved["underexposed"]:
                cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_underexposed.jpg"), img_under)
                sample_saved["underexposed"] = True

            # Overexposed
            img_over = apply_overexposure(base_img, severity)
            cv2.imwrite(os.path.join(split_dir, "overexposed", f"img_{i:04d}_over_s{severity}.jpg"), img_over)
            if not sample_saved["overexposed"]:
                cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_overexposed.jpg"), img_over)
                sample_saved["overexposed"] = True

            # Noise
            img_noise = apply_noise(base_img, severity)
            cv2.imwrite(os.path.join(split_dir, "noise", f"img_{i:04d}_noise_s{severity}.jpg"), img_noise)
            if not sample_saved["noise"]:
                cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_noise.jpg"), img_noise)
                sample_saved["noise"] = True

            # Corrupted
            img_corrupt = apply_corruption(base_img, severity)
            cv2.imwrite(os.path.join(split_dir, "corrupted", f"img_{i:04d}_corrupt_s{severity}.jpg"), img_corrupt)
            if not sample_saved["corrupted"]:
                cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_corrupted.jpg"), img_corrupt)
                sample_saved["corrupted"] = True

        item_id += 1

    print("Dataset generation complete!")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Sample images saved in: {SAMPLE_DIR}")

if __name__ == "__main__":
    generate_dataset(num_clean_base=100, train_ratio=0.8)
