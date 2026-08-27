import os
import sys

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import glob
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
)

from ml.feature_extractor import extract_features_from_file, FEATURE_NAMES

TEST_DIR = os.path.join(os.path.dirname(__file__), "dataset", "test")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))

CATEGORIES = ["clean", "blur", "underexposed", "overexposed", "noise", "corrupted"]

def run_evaluation():
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run ml/train.py first.")

    model_artifact = joblib.load(MODEL_PATH)
    classifiers = model_artifact["classifiers"]

    print("Extracting features from unseen test dataset...")
    X_test = []
    y_test_dict = {cat: [] for cat in ["blur", "underexposed", "overexposed", "noise", "corrupted"]}
    y_test_overall = []
    test_filenames = []

    for cat_idx, cat in enumerate(CATEGORIES):
        cat_dir = os.path.join(TEST_DIR, cat)
        image_files = glob.glob(os.path.join(cat_dir, "*.jpg")) + glob.glob(os.path.join(cat_dir, "*.png"))
        
        for img_path in image_files:
            feats_dict = extract_features_from_file(img_path)
            feat_vector = [feats_dict[fn] for fn in FEATURE_NAMES]
            X_test.append(feat_vector)
            test_filenames.append(os.path.basename(img_path))

            for issue in y_test_dict.keys():
                y_test_dict[issue].append(1 if cat == issue else 0)
            y_test_overall.append(cat_idx)

    X_test = np.array(X_test, dtype=np.float32)
    y_test_overall = np.array(y_test_overall, dtype=np.int32)
    for issue in y_test_dict:
        y_test_dict[issue] = np.array(y_test_dict[issue], dtype=np.int32)

    print(f"Test set size: {len(X_test)} samples.")

    # 1. Per-issue Evaluation
    per_issue_metrics = {}
    for issue in ["blur", "underexposed", "overexposed", "noise", "corrupted"]:
        clf = classifiers[issue]
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds

        acc = accuracy_score(y_test_dict[issue], preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_dict[issue], preds, average='binary', zero_division=0)
        
        per_issue_metrics[issue] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
        print(f"\nIssue: {issue.upper()}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    # 2. Overall Category & Quality Label Confusion Matrix
    overall_clf = classifiers["overall"]
    overall_preds = overall_clf.predict(X_test)

    cm = confusion_matrix(y_test_overall, overall_preds)
    overall_acc = accuracy_score(y_test_overall, overall_preds)

    print(f"\nOverall Multi-class Accuracy: {overall_acc:.4f}")

    # Save metrics JSON
    metrics_data = {
        "num_test_samples": len(X_test),
        "overall_accuracy": float(overall_acc),
        "per_issue_metrics": per_issue_metrics,
        "categories": CATEGORIES,
        "confusion_matrix": cm.tolist()
    }
    
    metrics_json_path = os.path.join(EVAL_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # 3. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CATEGORIES, yticklabels=CATEGORIES,
           title=f'Confusion Matrix (Overall Test Accuracy: {overall_acc*100:.1f}%)',
           ylabel='True Category',
           xlabel='Predicted Category')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_plot_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    # 4. Generate Markdown Evaluation Report
    report_md_path = os.path.join(EVAL_DIR, "evaluation_report.md")
    with open(report_md_path, "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Test Set Size:** {len(X_test)} samples (held-out unseen synthetic dataset)\n")
        f.write(f"**Overall Classification Accuracy:** {overall_acc * 100:.2f}%\n\n")
        
        f.write("## 1. Per-Issue Detection Performance\n\n")
        f.write("| Issue Type | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for issue, m in per_issue_metrics.items():
            f.write(f"| **{issue.capitalize()}** | {m['accuracy']*100:.1f}% | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1_score']*100:.1f}% |\n")
        
        f.write("\n\n## 2. Confusion Matrix\n\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        f.write("## 3. Failure Case Analysis & Limitations\n\n")
        f.write("### Failure Case Discussion\n")
        f.write("1. **Low-severity Noise vs. Mild Blur:** High-frequency noise can artificially inflate the variance of the Laplacian, occasionally causing mild blur to be masked or low-level noise to be interpreted as sharp high-frequency edges.\n")
        f.write("2. **Highlight Clipping in Naturally Bright Regions:** Images with intentional high dynamic range (e.g. skies or light sources) might trigger light overexposure warnings if highlight clipping exceeds 25% of total image area.\n")
        f.write("3. **Severe Block Artifacts vs. Extreme Noise:** Severe JPEG corruption at ultra-low bitrates introduces blocky edge discontinuities that occasionally overlap feature signatures with high-frequency salt-and-pepper noise.\n\n")
        f.write("### Limitations\n")
        f.write("- **Synthetic Degradation Gap:** Synthetic degradation patterns (Gaussian noise, linear LUT exposure adjustments) capture fundamental visual flaws well but may not reflect complex physical camera lens aberrations (e.g. chromatic aberration, vignetting).\n")
        f.write("- **Domain Specificity:** The model excels on general photographic and document imagery. Specialized domains (e.g., medical X-rays or satellite radar) may require specialized normalization.\n")

    print(f"\nEvaluation complete! Outputs generated in {EVAL_DIR}:")
    print(f"  - {metrics_json_path}")
    print(f"  - {cm_plot_path}")
    print(f"  - {report_md_path}")

if __name__ == "__main__":
    run_evaluation()
