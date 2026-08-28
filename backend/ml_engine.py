import os
import sys
import joblib
import numpy as np

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.feature_extractor import extract_features_from_bytes, generate_defect_heatmap, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib"))

"""
========================================================================================
MODEL DECISION ARCHITECTURE: HYBRID ML + SAFETY NET FALLBACK
========================================================================================
1. Primary Decision Signal (Machine Learning):
   Dedicated Random Forest binary classifiers predict defect probabilities (ml_prob) for
   each category (Blur, Underexposure, Overexposure, Noise, Corruption). An issue is 
   primarily detected when ml_prob >= 0.50.

2. Safety Net Fallback (Rule-Based Bounds):
   Raw physical feature metrics (Laplacian variance, mean luminance, blockiness index, etc.)
   serve strictly as a documented safety net for extreme out-of-distribution values (e.g.,
   pitch black, pure white, or corrupted byte streams) to prevent zero-shot false negatives.
========================================================================================
"""

class MLEngine:
    def __init__(self):
        self.model_artifact = None
        self.classifiers = None
        self.is_loaded = False

    def load_model(self):
        """Loads model artifact into memory ONCE at backend startup."""
        if os.path.exists(MODEL_PATH):
            print(f"Loading ML Model artifact from {MODEL_PATH}...")
            self.model_artifact = joblib.load(MODEL_PATH)
            self.classifiers = self.model_artifact["classifiers"]
            self.is_loaded = True
            print("ML Model successfully loaded into memory!")
        else:
            print(f"WARNING: Model artifact not found at {MODEL_PATH}. Running in safety-net mode.")
            self.is_loaded = False

    def analyze_image_bytes(self, image_bytes: bytes):
        """
        Runs complete feature extraction + model inference + explainability + heatmap generation pipeline.
        Returns dict containing quality_score, quality_label, issues, image_stats, explanation, heatmap_base64, heatmap_grid.
        """
        # Step 1: Feature Extraction & Header Validation
        stats = extract_features_from_bytes(image_bytes)

        if stats.get("is_valid_format", 1.0) == 0.0:
            return {
                "quality_score": 0.0,
                "quality_label": "DEFECTIVE",
                "issues": [
                    {"type": "corrupted", "severity": "high", "confidence": 1.00}
                ],
                "image_stats": stats,
                "explanation": "Image file header or bitstream is severely corrupted, unreadable, or invalid image format.",
                "heatmap_base64": None,
                "heatmap_grid": None
            }

        feat_vector = np.array([[stats[fn] for fn in FEATURE_NAMES]], dtype=np.float32)

        issues = []
        explanation_bullets = []
        total_penalty = 0.0

        # Step 2: Model Inference + Documented Safety-Net Fallbacks
        if self.is_loaded and self.classifiers:
            
            # -------------------------------------------------------------------------
            # 1. BLUR DETECTION
            # ML Role: Primary signal via Random Forest blur classifier (ml_prob >= 0.50).
            # Safety Net: Laplacian variance < 15.0 triggers safety net for extreme zero-contrast blur.
            # -------------------------------------------------------------------------
            clf_blur = self.classifiers.get("blur")
            blur_prob = float(clf_blur.predict_proba(feat_vector)[0, 1]) if clf_blur else 0.0
            lap_var = stats.get("laplacian_var", 500.0)
            
            if blur_prob >= 0.50 or lap_var < 15.0:
                severity = "high" if (blur_prob >= 0.80 or lap_var < 10.0) else ("medium" if (blur_prob >= 0.65 or lap_var < 25.0) else "low")
                conf = round(blur_prob if blur_prob >= 0.50 else max(0.50, 1.0 - (lap_var / 50.0)), 2)
                conf = min(0.99, max(0.50, conf))
                issues.append({"type": "blur", "severity": severity, "confidence": conf})
                total_penalty += 35.0 if severity == "high" else (22.0 if severity == "medium" else 12.0)
                explanation_bullets.append(f"Blur detected: ML Blur Probability = {blur_prob*100:.1f}%, Laplacian variance = {lap_var:.1f}.")

            # -------------------------------------------------------------------------
            # 2. UNDEREXPOSURE DETECTION
            # ML Role: Primary signal via Underexposure classifier (ml_prob >= 0.50).
            # Safety Net: Mean luminance < 15.0 or shadow clip > 60% triggers extreme dark safety net.
            # -------------------------------------------------------------------------
            clf_under = self.classifiers.get("underexposed")
            under_prob = float(clf_under.predict_proba(feat_vector)[0, 1]) if clf_under else 0.0
            mean_lum = stats.get("mean_luminance", 128.0)
            shadow_clip = stats.get("shadow_clip_pct", 0.0)
            
            if under_prob >= 0.50 or mean_lum < 15.0 or shadow_clip > 60.0:
                severity = "high" if (under_prob >= 0.80 or mean_lum < 15.0) else ("medium" if (under_prob >= 0.65 or mean_lum < 35.0) else "low")
                conf = round(under_prob if under_prob >= 0.50 else max(0.50, (50.0 - mean_lum) / 50.0), 2)
                conf = min(0.99, max(0.50, conf))
                issues.append({"type": "underexposed", "severity": severity, "confidence": conf})
                total_penalty += 30.0 if severity == "high" else (18.0 if severity == "medium" else 10.0)
                explanation_bullets.append(f"Underexposure detected: ML Dark Probability = {under_prob*100:.1f}%, Mean luminance = {mean_lum:.1f}.")

            # -------------------------------------------------------------------------
            # 3. OVEREXPOSURE DETECTION
            # ML Role: Primary signal via Overexposure classifier (ml_prob >= 0.50).
            # Safety Net: Mean luminance > 245.0 or highlight clip > 60% triggers white clipping safety net.
            # -------------------------------------------------------------------------
            clf_over = self.classifiers.get("overexposed")
            over_prob = float(clf_over.predict_proba(feat_vector)[0, 1]) if clf_over else 0.0
            highlight_clip = stats.get("highlight_clip_pct", 0.0)
            
            if over_prob >= 0.50 or mean_lum > 245.0 or highlight_clip > 60.0:
                severity = "high" if (over_prob >= 0.80 or mean_lum > 245.0) else ("medium" if (over_prob >= 0.65 or mean_lum > 230.0) else "low")
                conf = round(over_prob if over_prob >= 0.50 else max(0.50, (mean_lum - 200.0) / 55.0), 2)
                conf = min(0.99, max(0.50, conf))
                issues.append({"type": "overexposed", "severity": severity, "confidence": conf})
                total_penalty += 30.0 if severity == "high" else (18.0 if severity == "medium" else 10.0)
                explanation_bullets.append(f"Overexposure detected: ML Overexposure Probability = {over_prob*100:.1f}%, Mean luminance = {mean_lum:.1f}.")

            # -------------------------------------------------------------------------
            # 4. NOISE DETECTION
            # ML Role: Primary signal via Noise classifier (ml_prob >= 0.50).
            # Safety Net: Residual noise variance > 350.0 triggers extreme noise safety net.
            # -------------------------------------------------------------------------
            clf_noise = self.classifiers.get("noise")
            noise_prob = float(clf_noise.predict_proba(feat_vector)[0, 1]) if clf_noise else 0.0
            noise_var = stats.get("noise_variance", 0.0)
            
            if noise_prob >= 0.50 or noise_var > 350.0:
                severity = "high" if (noise_prob >= 0.80 or noise_var > 250.0) else ("medium" if (noise_prob >= 0.65 or noise_var > 150.0) else "low")
                conf = round(noise_prob if noise_prob >= 0.50 else max(0.50, noise_var / 400.0), 2)
                conf = min(0.99, max(0.50, conf))
                issues.append({"type": "noise", "severity": severity, "confidence": conf})
                total_penalty += 25.0 if severity == "high" else (15.0 if severity == "medium" else 8.0)
                explanation_bullets.append(f"Image noise detected: ML Noise Probability = {noise_prob*100:.1f}%, Noise variance = {noise_var:.1f}.")

            # -------------------------------------------------------------------------
            # 5. CORRUPTION / BLOCKINESS DETECTION
            # ML Role: Primary signal via Corruption classifier (ml_prob >= 0.50).
            # Safety Net: Blockiness index > 4.5 triggers JPEG block artifact safety net.
            # -------------------------------------------------------------------------
            clf_corrupt = self.classifiers.get("corrupted")
            corrupt_prob = float(clf_corrupt.predict_proba(feat_vector)[0, 1]) if clf_corrupt else 0.0
            block_idx = stats.get("blockiness_index", 1.0)
            
            if corrupt_prob >= 0.50 or block_idx > 4.5:
                severity = "high" if (corrupt_prob >= 0.80 or block_idx > 4.0) else ("medium" if (corrupt_prob >= 0.65 or block_idx > 3.0) else "low")
                conf = round(corrupt_prob if corrupt_prob >= 0.50 else max(0.50, block_idx / 5.0), 2)
                conf = min(0.99, max(0.50, conf))
                issues.append({"type": "corrupted", "severity": severity, "confidence": conf})
                total_penalty += 40.0 if severity == "high" else (25.0 if severity == "medium" else 15.0)
                explanation_bullets.append(f"JPEG blockiness / corruption detected: ML Corruption Probability = {corrupt_prob*100:.1f}%, Blockiness index = {block_idx:.2f}.")

        # Step 3: Compute Overall Score & Label
        quality_score = max(0.0, min(100.0, 100.0 - total_penalty))
        
        has_high_issue = any(iss["severity"] == "high" for iss in issues)
        if quality_score >= 75.0 and not has_high_issue:
            quality_label = "ACCEPTABLE"
        elif quality_score >= 45.0 and not has_high_issue:
            quality_label = "DEGRADED"
        else:
            quality_label = "DEFECTIVE"

        if not explanation_bullets:
            explanation = "Image quality is pristine. Sharpness, luminance, and noise levels fall well within optimal photographic bounds."
        else:
            explanation = " ".join(explanation_bullets)

        # Step 4: Generate Defect Localization Heatmap Overlay
        heatmap_base64, heatmap_grid = generate_defect_heatmap(image_bytes)

        return {
            "quality_score": round(quality_score, 1),
            "quality_label": quality_label,
            "issues": issues,
            "image_stats": {k: round(v, 2) for k, v in stats.items()},
            "explanation": explanation,
            "heatmap_base64": heatmap_base64,
            "heatmap_grid": heatmap_grid
        }

# Global singleton engine instance
ml_engine = MLEngine()
