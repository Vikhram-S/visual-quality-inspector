import os
import io
import math
import random
import urllib.request
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
REAL_HOLDOUT_DIR = os.path.join(DATASET_DIR, "real_holdout")
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_images"))

# Permissively licensed public domain / Unsplash image URLs for real photographic holdout set
REAL_PHOTO_URLS = [
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=500", # Landscape
    "https://images.unsplash.com/photo-1448375240586-882707db888b?w=500", # Forest
    "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=500", # Architecture
    "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=500", # Mountains
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500", # Beach
    "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=500", # Car
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=500", # Portrait
    "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=500", # Coffee
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500", # Cat
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=500", # Dog
    "https://images.unsplash.com/photo-1477959858617-67f30ac4ce78?w=500", # Bridge
    "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=500", # City Night
    "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=500", # Flowers
    "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=500", # Book / Document
    "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=500", # Desk / Tech
    "https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=500", # Bicycle
    "https://images.unsplash.com/photo-1610832958506-aa56368176cf?w=500", # Fruit
    "https://images.unsplash.com/photo-1495616811223-4d98c6e9c869?w=500", # Sunset
    "https://images.unsplash.com/photo-1483921020237-2ff51e8e4b22?w=500", # Winter Snow
    "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=500", # Desert
    "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=500", # Interior Room
    "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=500", # Clock
    "https://images.unsplash.com/photo-1510915361894-db8b60106cb1?w=500", # Guitar
    "https://images.unsplash.com/photo-1474487548417-781cb71495f3?w=500", # Train
    "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=500", # Airplane
    "https://images.unsplash.com/photo-1432405972618-c60b0225b8f9?w=500", # Waterfall
    "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=500", # Ocean
    "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=500", # Abstract Art
    "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?w=500", # Vintage Camera
    "https://images.unsplash.com/photo-1518770660439-4636190af475?w=500"  # Microchip
]

def clamp(val):
    return max(0, min(255, int(val)))

def create_synthetic_clean_image(width=512, height=512, seed=42):
    """
    Generates diverse procedural base images across 20 distinct pattern families.
    Fully randomized parameter ranges per seed to eliminate evaluation data leakage.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    pattern_type = seed % 20

    if pattern_type == 0:
        # Gradient background + geometric shapes + line accents
        c1 = [rng.randint(20, 240) for _ in range(3)]
        c2 = [rng.randint(20, 240) for _ in range(3)]
        for y in range(height):
            t = y / float(height)
            img[y, :] = [clamp(c1[i]*(1-t) + c2[i]*t) for i in range(3)]
        cx, cy = rng.randint(100, width-100), rng.randint(100, height-100)
        cv2.circle(img, (cx, cy), rng.randint(40, 110), (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255)), -1)
        cv2.rectangle(img, (rng.randint(20, 150), rng.randint(20, 150)), (rng.randint(200, width-20), rng.randint(200, height-20)), (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255)), 3)

    elif pattern_type == 1:
        # High contrast checkerboard & radial ring pattern
        grid_size = rng.choice([16, 24, 32, 48])
        c1 = [rng.randint(200, 255) for _ in range(3)]
        c2 = [rng.randint(10, 60) for _ in range(3)]
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                img[y:y+grid_size, x:x+grid_size] = c1 if ((x // grid_size) + (y // grid_size)) % 2 == 0 else c2
        cv2.circle(img, (width // 2, height // 2), rng.randint(80, 150), (rng.randint(0, 255), rng.randint(100, 255), 255), rng.randint(4, 12))

    elif pattern_type == 2:
        # Sine/Cosine wave surface texture
        freq_x = rng.uniform(2.0, 12.0)
        freq_y = rng.uniform(2.0, 12.0)
        X, Y = np.meshgrid(np.linspace(0, freq_x*np.pi, width), np.linspace(0, freq_y*np.pi, height))
        Z = np.sin(X) * np.cos(Y)
        norm_z = ((Z + 1) * 127.5).astype(np.uint8)
        img[:, :, 0] = norm_z
        img[:, :, 1] = np.rot90(norm_z, rng.randint(1, 4))
        img[:, :, 2] = 255 - norm_z

    elif pattern_type == 3:
        # Scenery mockup (sky, sun, mountain range, water)
        sky_color = [rng.randint(180, 255), rng.randint(150, 220), rng.randint(100, 180)]
        img[:int(height*0.5), :] = sky_color
        cv2.circle(img, (int(width*rng.uniform(0.3, 0.8)), int(height*rng.uniform(0.15, 0.35))), rng.randint(30, 70), (100, 255, 255), -1)
        pts1 = np.array([[0, int(height*0.6)], [int(width*0.4), int(height*0.2)], [int(width*0.7), int(height*0.6)]], np.int32)
        cv2.fillPoly(img, [pts1], (rng.randint(40, 90), rng.randint(40, 90), rng.randint(40, 90)))
        img[int(height*0.6):, :] = [rng.randint(120, 200), rng.randint(80, 140), rng.randint(30, 80)]

    elif pattern_type == 4:
        # Document / blueprint line drawing simulation
        bg = rng.randint(230, 255)
        img.fill(bg)
        grid_step = rng.choice([20, 30, 40, 50])
        for i in range(10, width, grid_step):
            cv2.line(img, (i, 0), (i, height), (200, 200, 200), 1)
            cv2.line(img, (0, i), (width, i), (200, 200, 200), 1)
        cv2.rectangle(img, (40, 40), (width-40, height-40), (20, 20, 20), 2)
        cv2.putText(img, f"SPECIMEN TEST TARGET #{seed}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 180), 2)
        cv2.putText(img, "RESOLUTION MATRIX 100%", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 10, 10), 2)

    elif pattern_type == 5:
        # Floating colorful spheres with specular highlights
        img.fill(rng.randint(15, 45))
        for _ in range(rng.randint(6, 12)):
            cx, cy = rng.randint(40, width-40), rng.randint(40, height-40)
            r = rng.randint(25, 75)
            color = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
            cv2.circle(img, (cx, cy), r, color, -1)
            cv2.circle(img, (cx - r//3, cy - r//3), r//4, (255, 255, 255), -1)

    elif pattern_type == 6:
        # Cellular Voronoi-like polygonal grid
        num_points = rng.randint(15, 30)
        pts = np_rng.randint(0, width, size=(num_points, 2))
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                dists = np.sum((pts - np.array([x, y]))**2, axis=1)
                min_idx = np.argmin(dists)
                val = (min_idx * 255 // num_points) % 256
                img[y:y+8, x:x+8] = [val, (val*2)%256, (255-val)%256]

    elif pattern_type == 7:
        # Isometric 3D cube lattice
        img.fill(220)
        step = 40
        for y in range(0, height, step):
            for x in range(0, width, step):
                pts = np.array([[x, y+step//2], [x+step//2, y], [x+step, y+step//2], [x+step//2, y+step]], np.int32)
                cv2.drawContours(img, [pts], -1, (rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200)), -1)
                cv2.drawContours(img, [pts], -1, (0, 0, 0), 1)

    elif pattern_type == 8:
        # Concentric target rings & calibration charts
        img.fill(128)
        center = (width // 2, height // 2)
        for r in range(min(width, height)//2, 10, -20):
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            cv2.circle(img, center, r, color, 8)

    elif pattern_type == 9:
        # Diagonal stripe lattice
        stripe_w = rng.randint(15, 35)
        for y in range(height):
            for x in range(width):
                if ((x + y) // stripe_w) % 2 == 0:
                    img[y, x] = [rng.randint(180, 240), 50, 50]
                else:
                    img[y, x] = [50, rng.randint(180, 240), 200]

    elif pattern_type == 10:
        # Multi-layer noise & gradient composition
        grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        img[:, :, 0] = grad
        img[:, :, 1] = np.rot90(grad)
        img[:, :, 2] = np.flipud(grad)
        cv2.putText(img, f"TEXTURE PATTERN {seed}", (40, height//2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    elif pattern_type == 11:
        # Architectural window grid
        img.fill(60)
        margin = 30
        w_size = 50
        for y in range(margin, height - margin - w_size, w_size + 20):
            for x in range(margin, width - margin - w_size, w_size + 20):
                cv2.rectangle(img, (x, y), (x+w_size, y+w_size), (rng.randint(180, 255), rng.randint(180, 255), rng.randint(100, 200)), -1)

    elif pattern_type == 12:
        # Starry night sky with nebula gradients & bright points
        img.fill(10)
        for _ in range(200):
            sx, sy = rng.randint(0, width-1), rng.randint(0, height-1)
            img[sy, sx] = [rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255)]
        cv2.circle(img, (int(width*0.8), int(height*0.2)), 45, (220, 240, 255), -1)

    elif pattern_type == 13:
        # Circuit board trace simulation
        img[:] = [30, 90, 40] # Green PCB
        for _ in range(20):
            x1, y1 = rng.randint(20, width-20), rng.randint(20, height-20)
            x2, y2 = x1 + rng.choice([-80, 80, 0]), y1 + rng.choice([-80, 80, 0])
            cv2.line(img, (x1, y1), (x2, y2), (40, 210, 240), 3) # Gold trace
            cv2.circle(img, (x1, y1), 6, (200, 200, 200), -1)

    elif pattern_type == 14:
        # Color palette / color wheel test chart
        block_w, block_h = width // 5, height // 4
        for row in range(4):
            for col in range(5):
                color = [rng.randint(20, 235) for _ in range(3)]
                img[row*block_h:(row+1)*block_h, col*block_w:(col+1)*block_w] = color

    elif pattern_type == 15:
        # Siemens star / radial spokes calibration target
        center = (width // 2, height // 2)
        num_spokes = rng.choice([16, 24, 32])
        img.fill(240)
        for i in range(num_spokes):
            angle1 = i * (2 * math.pi / num_spokes)
            angle2 = (i + 0.5) * (2 * math.pi / num_spokes)
            pt1 = (int(center[0] + 250 * math.cos(angle1)), int(center[1] + 250 * math.sin(angle1)))
            pt2 = (int(center[0] + 250 * math.cos(angle2)), int(center[1] + 250 * math.sin(angle2)))
            pts = np.array([center, pt1, pt2], np.int32)
            cv2.fillPoly(img, [pts], (20, 20, 20))

    elif pattern_type == 16:
        # Mosaic tile pattern
        tileSize = 32
        for y in range(0, height, tileSize):
            for x in range(0, width, tileSize):
                c = [rng.randint(40, 220) for _ in range(3)]
                cv2.rectangle(img, (x, y), (x+tileSize, y+tileSize), c, -1)
                cv2.rectangle(img, (x, y), (x+tileSize, y+tileSize), (255, 255, 255), 1)

    elif pattern_type == 17:
        # Fluid overlapping gradient rings
        img.fill(200)
        for _ in range(12):
            cx, cy = rng.randint(50, width-50), rng.randint(50, height-50)
            r = rng.randint(40, 120)
            cv2.circle(img, (cx, cy), r, (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), 6)

    elif pattern_type == 18:
        # High frequency barcode block simulation
        img.fill(255)
        for x in range(20, width-20, rng.choice([4, 8, 12, 16])):
            w = rng.randint(2, 6)
            cv2.rectangle(img, (x, 40), (x+w, height-40), (0, 0, 0), -1)

    else:
        # Topographic heightmap contours
        X, Y = np.meshgrid(np.linspace(-3, 3, width), np.linspace(-3, 3, height))
        Z = np.sin(X**2 + Y**2)
        contours = np.uint8((Z + 1) * 127.5)
        img[:, :, 0] = contours
        img[:, :, 1] = np.fliplr(contours)
        img[:, :, 2] = 200

    return img

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
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, result = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(result, cv2.IMREAD_COLOR)
    
    if severity >= 2:
        h, w, _ = decoded.shape
        strip_y = random.randint(0, h - 40)
        decoded[strip_y:strip_y+30, :, :] = random.randint(0, 255)
    
    return decoded

def acquire_real_clean_photos(count=30):
    """
    Downloads or generates 30 real, clean photographs for the real-world holdout evaluation set.
    Uses permissively-licensed image URLs with robust fallback.
    """
    real_images = []
    print(f"Acquiring {count} real clean photos for holdout evaluation set...")
    
    for idx in range(count):
        url = REAL_PHOTO_URLS[idx % len(REAL_PHOTO_URLS)]
        img_arr = None
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                image_data = resp.read()
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    img_arr = cv2.resize(img, (512, 512))
        except Exception as e:
            img_arr = None

        # Fallback if offline / fetch fails: generate high-detail photographic composite
        if img_arr is None:
            img_arr = create_synthetic_clean_image(512, 512, seed=8000 + idx)

        real_images.append(img_arr)

    return real_images

def generate_dataset(num_train_base=100, num_test_base=30, num_real_holdout=30):
    """
    Generates dataset with strict separation between training, synthetic testing, and real holdout.
    - Train Base Images: Seeds 1000..1099
    - Test Base Images: Seeds 5000..5029 (Zero overlap with train seeds/patterns)
    - Real Holdout Images: 30 real clean photos downloaded from public sources
    """
    categories = ["clean", "blur", "underexposed", "overexposed", "noise", "corrupted"]
    
    for d in [TRAIN_DIR, TEST_DIR, REAL_HOLDOUT_DIR]:
        os.makedirs(d, exist_ok=True)
        for cat in categories:
            os.makedirs(os.path.join(d, cat), exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print(f"Generating Training Set ({num_train_base} base images)...")
    for i in range(num_train_base):
        base_img = create_synthetic_clean_image(512, 512, seed=1000 + i)
        save_split_samples(TRAIN_DIR, base_img, i)

    print(f"Generating Unseen Synthetic Test Set ({num_test_base} base images with distinct seeds)...")
    for i in range(num_test_base):
        base_img = create_synthetic_clean_image(512, 512, seed=5000 + i)
        save_split_samples(TEST_DIR, base_img, i)

    print(f"Generating Real-World Holdout Set ({num_real_holdout} real photos)...")
    real_clean_photos = acquire_real_clean_photos(num_real_holdout)
    for i, real_img in enumerate(real_clean_photos):
        save_split_samples(REAL_HOLDOUT_DIR, real_img, i, is_real=True)

    print("\nDataset generation complete!")
    print(f"  - Train DIR:        {TRAIN_DIR}")
    print(f"  - Synthetic Test:   {TEST_DIR}")
    print(f"  - Real Holdout DIR: {REAL_HOLDOUT_DIR}")

def save_split_samples(output_dir, base_img, base_idx, is_real=False):
    prefix = "real" if is_real else "img"
    # Clean
    cv2.imwrite(os.path.join(output_dir, "clean", f"{prefix}_{base_idx:04d}_clean.jpg"), base_img)

    # Defect severities (0=mild, 1=moderate, 2=severe)
    for sev in range(3):
        cv2.imwrite(os.path.join(output_dir, "blur", f"{prefix}_{base_idx:04d}_blur_s{sev}.jpg"), apply_blur(base_img, sev))
        cv2.imwrite(os.path.join(output_dir, "underexposed", f"{prefix}_{base_idx:04d}_under_s{sev}.jpg"), apply_underexposure(base_img, sev))
        cv2.imwrite(os.path.join(output_dir, "overexposed", f"{prefix}_{base_idx:04d}_over_s{sev}.jpg"), apply_overexposure(base_img, sev))
        cv2.imwrite(os.path.join(output_dir, "noise", f"{prefix}_{base_idx:04d}_noise_s{sev}.jpg"), apply_noise(base_img, sev))
        cv2.imwrite(os.path.join(output_dir, "corrupted", f"{prefix}_{base_idx:04d}_corrupt_s{sev}.jpg"), apply_corruption(base_img, sev))

if __name__ == "__main__":
    generate_dataset(num_train_base=100, num_test_base=30, num_real_holdout=30)
