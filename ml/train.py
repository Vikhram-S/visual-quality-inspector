import os
import sys

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import glob
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

from ml.feature_extractor import extract_features_from_file, FEATURE_NAMES

TRAIN_DIR = os.path.join(os.path.dirname(__file__), "dataset", "train")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
BACKEND_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "model.joblib"))

CATEGORIES = ["clean", "blur", "underexposed", "overexposed", "noise", "corrupted"]

def build_training_dataset():
    """
    Loads all training images, extracts features, and builds multi-label binary target vectors.
    Targets for each sample: [is_blur, is_underexposed, is_overexposed, is_noise, is_corrupted]
    """
    X = []
    y_blur = []
    y_under = []
    y_over = []
    y_noise = []
    y_corrupt = []
    y_overall_class = [] # 0: clean, 1: blur, 2: under, 3: over, 4: noise, 5: corrupt

    print("Extracting features from training dataset...")
    
    for cat_idx, cat in enumerate(CATEGORIES):
        cat_dir = os.path.join(TRAIN_DIR, cat)
        image_files = glob.glob(os.path.join(cat_dir, "*.jpg")) + glob.glob(os.path.join(cat_dir, "*.png"))
        
        print(f"  Category '{cat}': {len(image_files)} images")
        for img_path in image_files:
            feats_dict = extract_features_from_file(img_path)
            feat_vector = [feats_dict[fn] for fn in FEATURE_NAMES]
            X.append(feat_vector)
            
            y_blur.append(1 if cat == "blur" else 0)
            y_under.append(1 if cat == "underexposed" else 0)
            y_over.append(1 if cat == "overexposed" else 0)
            y_noise.append(1 if cat == "noise" else 0)
            y_corrupt.append(1 if cat == "corrupted" else 0)
            y_overall_class.append(cat_idx)

    X = np.array(X, dtype=np.float32)
    
    targets = {
        "blur": np.array(y_blur, dtype=np.int32),
        "underexposed": np.array(y_under, dtype=np.int32),
        "overexposed": np.array(y_over, dtype=np.int32),
        "noise": np.array(y_noise, dtype=np.int32),
        "corrupted": np.array(y_corrupt, dtype=np.int32),
        "overall_class": np.array(y_overall_class, dtype=np.int32)
    }

    return X, targets

def train_and_save_model():
    X_train, y_targets = build_training_dataset()
    
    print(f"\nTraining set size: {X_train.shape[0]} samples, {X_train.shape[1]} features.")

    # We train dedicated RandomForest classifiers for each issue type
    classifiers = {}
    feature_importances = {}

    for issue_type in ["blur", "underexposed", "overexposed", "noise", "corrupted"]:
        print(f"Training classifier for '{issue_type}'...")
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ])
        pipe.fit(X_train, y_targets[issue_type])
        
        # Evaluate on train
        preds = pipe.predict(X_train)
        f1 = f1_score(y_targets[issue_type], preds, zero_division=0)
        print(f"  '{issue_type}' Train F1 Score: {f1:.4f}")
        
        classifiers[issue_type] = pipe
        
        # Extract feature importances
        rf_model = pipe.named_steps['clf']
        importances = rf_model.feature_importances_
        feature_importances[issue_type] = {
            FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))
        }

    # Overall multi-class classifier for main label
    print("Training overall multi-class classifier...")
    overall_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=80, max_depth=6, random_state=42))
    ])
    overall_pipe.fit(X_train, y_targets["overall_class"])
    classifiers["overall"] = overall_pipe

    model_artifact = {
        "classifiers": classifiers,
        "feature_names": FEATURE_NAMES,
        "categories": CATEGORIES,
        "feature_importances": feature_importances
    }

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BACKEND_MODEL_PATH), exist_ok=True)

    joblib.dump(model_artifact, MODEL_SAVE_PATH)
    joblib.dump(model_artifact, BACKEND_MODEL_PATH)

    print(f"\nModel artifacts successfully saved to:")
    print(f"  - {MODEL_SAVE_PATH}")
    print(f"  - {BACKEND_MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
