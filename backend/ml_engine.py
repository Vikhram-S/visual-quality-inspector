import os
import sys
import joblib
import numpy as np

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.feature_extractor import extract_features_from_bytes, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib"))

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
            print(f"WARNING: Model artifact not found at {MODEL_PATH}. Running in rule-based fallback mode.")
            self.is_loaded = False

    def analyze_image_bytes(self, image_bytes: bytes):
        """
        Runs complete feature extraction + model inference + explainability pipeline.
        Returns dict containing quality_score, quality_label, issues, image_stats, explanation.
        """
        # Step 1: Feature Extraction
        stats = extract_features_from_bytes(image_bytes)

        # Check for immediate corruption / unreadable image file header
        if stats.get("is_valid_format", 1.0) == 0.0:
            return {
                "quality_score": 0.0,
                "quality_label": "DEFECTIVE",
                "issues": [
                    {"type": "corrupted", "severity": "high", "confidence": 1.00}
                ],
                "image_stats": stats,
                "explanation": "Image file header or bitstream is severely corrupted, unreadable, or invalid image format."
            }

        feat_vector = np.array([[stats[fn] for fn in FEATURE_NAMES]], dtype=np.float32)

        issues = []
        explanation_bullets = []
        total_penalty = 0.0

        # Step 2: Issue-specific classification + feature rules
        if self.is_loaded and self.classifiers:
            # 1. Blur Detection
            clf_blur = self.classifiers.get("blur")
            blur_prob = float(clf_blur.predict_proba(feat_vector)[0, 1]) if clf_blur else 0.0
            lap_var = stats.get("laplacian_var", 500.0)
            
            # Combine ML probability with domain feature thresholds
            if blur_prob >= 0.40 or lap_var < 80.0:
                severity = "high" if (blur_prob > 0.75 or lap_var < 35.0) else ("medium" if (blur_prob > 0.55 or lap_var < 60.0) else "low")
                conf = round(max(blur_prob, 1.0 - (lap_var / 120.0)), 2)
                conf = min(0.99, max(0.40, conf))
                issues.append({"type": "blur", "severity": severity, "confidence": conf})
                total_penalty += 35.0 if severity == "high" else (22.0 if severity == "medium" else 12.0)
                explanation_bullets.append(f"Blur detected: Laplacian variance = {lap_var:.1f} (below optimal sharpness threshold 100.0).")

            # 2. Underexposure Detection
            clf_under = self.classifiers.get("underexposed")
            under_prob = float(clf_under.predict_proba(feat_vector)[0, 1]) if clf_under else 0.0
            mean_lum = stats.get("mean_luminance", 128.0)
            shadow_clip = stats.get("shadow_clip_pct", 0.0)
            
            if under_prob >= 0.40 or mean_lum < 60.0 or shadow_clip > 20.0:
                severity = "high" if (under_prob > 0.75 or mean_lum < 35.0 or shadow_clip > 45.0) else ("medium" if (under_prob > 0.55 or mean_lum < 50.0) else "low")
                conf = round(max(under_prob, min(0.99, (80.0 - mean_lum) / 80.0)), 2)
                conf = min(0.99, max(0.40, conf))
                issues.append({"type": "underexposed", "severity": severity, "confidence": conf})
                total_penalty += 30.0 if severity == "high" else (18.0 if severity == "medium" else 10.0)
                explanation_bullets.append(f"Underexposure detected: Mean luminance = {mean_lum:.1f} (optimal 80-180), {shadow_clip:.1f}% shadow clipping.")

            # 3. Overexposure Detection
            clf_over = self.classifiers.get("overexposed")
            over_prob = float(clf_over.predict_proba(feat_vector)[0, 1]) if clf_over else 0.0
            highlight_clip = stats.get("highlight_clip_pct", 0.0)
            
            if over_prob >= 0.40 or mean_lum > 210.0 or highlight_clip > 20.0:
                severity = "high" if (over_prob > 0.75 or mean_lum > 235.0 or highlight_clip > 40.0) else ("medium" if (over_prob > 0.55 or mean_lum > 220.0) else "low")
                conf = round(max(over_prob, min(0.99, (mean_lum - 190.0) / 65.0)), 2)
                conf = min(0.99, max(0.40, conf))
                issues.append({"type": "overexposed", "severity": severity, "confidence": conf})
                total_penalty += 30.0 if severity == "high" else (18.0 if severity == "medium" else 10.0)
                explanation_bullets.append(f"Overexposure detected: Mean luminance = {mean_lum:.1f}, {highlight_clip:.1f}% highlight clipping.")

            # 4. Noise Detection
            clf_noise = self.classifiers.get("noise")
            noise_prob = float(clf_noise.predict_proba(feat_vector)[0, 1]) if clf_noise else 0.0
            noise_var = stats.get("noise_variance", 0.0)
            
            if noise_prob >= 0.40 or noise_var > 60.0:
                severity = "high" if (noise_prob > 0.75 or noise_var > 200.0) else ("medium" if (noise_prob > 0.55 or noise_var > 110.0) else "low")
                conf = round(max(noise_prob, min(0.99, noise_var / 250.0)), 2)
                conf = min(0.99, max(0.40, conf))
                issues.append({"type": "noise", "severity": severity, "confidence": conf})
                total_penalty += 25.0 if severity == "high" else (15.0 if severity == "medium" else 8.0)
                explanation_bullets.append(f"Image noise detected: Residual high-frequency noise variance = {noise_var:.1f}.")

            # 5. Corruption / Blockiness
            clf_corrupt = self.classifiers.get("corrupted")
            corrupt_prob = float(clf_corrupt.predict_proba(feat_vector)[0, 1]) if clf_corrupt else 0.0
            block_idx = stats.get("blockiness_index", 1.0)
            
            if corrupt_prob >= 0.40 or block_idx > 2.2:
                severity = "high" if (corrupt_prob > 0.75 or block_idx > 3.5) else ("medium" if (corrupt_prob > 0.55 or block_idx > 2.8) else "low")
                conf = round(max(corrupt_prob, min(0.99, block_idx / 4.0)), 2)
                conf = min(0.99, max(0.40, conf))
                issues.append({"type": "corrupted", "severity": severity, "confidence": conf})
                total_penalty += 40.0 if severity == "high" else (25.0 if severity == "medium" else 15.0)
                explanation_bullets.append(f"JPEG blockiness / artifact corruption detected: Blockiness index = {block_idx:.2f}.")

        # Calculate final overall score
        quality_score = max(0.0, min(100.0, 100.0 - total_penalty))
        
        # Determine overall label
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

        return {
            "quality_score": round(quality_score, 1),
            "quality_label": quality_label,
            "issues": issues,
            "image_stats": {k: round(v, 2) for k, v in stats.items()},
            "explanation": explanation
        }

# Global singleton engine instance
ml_engine = MLEngine()
