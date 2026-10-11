# ============================================================
# Datasets: HF and VF
# Expected directory structure:
# dataset/
#   HF/
#     train_data/Normal, train_data/Violence
#     test_data/Normal,  test_data/Violence
#   VF/
#     train_data/Normal, train_data/Violence
#     test_data/Normal,  test_data/Violence
# ============================================================

import os
import random
import pickle
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, classification_report
)

# --------------------------- Configuration ---------------------------

SEED = 42
DATA_ROOT = Path("dataset")
RESULT_ROOT = Path("result")
DATASET_NAMES = ("HF", "VF")
CLASS_NAMES = ("Normal", "Violence")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Windows-friendly
CV_SPLITS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()
print(f"Device: {DEVICE}")


def make_output_dirs(base_dir):
    names = (
        "plots", "confusion", "roc", "pr_curve", "metrics",
        "results", "features", "pickle", "trained_models",
        "predictions", "cross_validation"
    )
    paths = {name: base_dir / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------- Dataset loading ---------------------------

class ViolenceImageDataset(Dataset):
    """Image dataset returning (image_tensor, label, filepath)."""

    def __init__(self, samples, transform=TRANSFORM):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Unable to read image: {path}") from exc
        return tensor, int(label), str(path)


def collect_samples(split_dir):
    """Collect samples in deterministic order; labels: Normal=0, Violence=1."""
    split_dir = Path(split_dir)
    samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required class directory not found: {class_dir}"
            )
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples.extend((str(path), label) for path in files)

    if not samples:
        raise FileNotFoundError(
            f"No supported images found in {split_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )
    return samples


def validate_split(samples, dataset_name, split_name):
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for _, label in samples:
        counts[CLASS_NAMES[int(label)]] += 1
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"{dataset_name}/{split_name} has no samples for class(es): "
            f"{', '.join(missing)}. Counts: {counts}"
        )
    print(f"{dataset_name} {split_name}: {len(samples)} images | {counts}")


# --------------------------- Model components ---------------------------

class FrequencyEnhancer(nn.Module):
    """Fixed depthwise Sobel-X filter for frequency/edge enhancement."""

    def __init__(self, channels):
        super().__init__()
        self.edge = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        kernel = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]],
            dtype=torch.float32
        ).unsqueeze(1)
        with torch.no_grad():
            self.edge.weight.copy_(kernel.repeat(channels, 1, 1, 1))
        self.edge.weight.requires_grad_(False)

    def forward(self, x):
        return self.edge(x)


class AttentionBlock(nn.Module):
    """Channel and spatial attention."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x = x * self.channel(x)
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.max(dim=1, keepdim=True).values
        spatial_map = torch.sigmoid(
            self.spatial(torch.cat((avg, maximum), dim=1))
        )
        return x * spatial_map


class TransformerBlock(nn.Module):
    def __init__(self, dim=1280, heads=8):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Transformer dimension must be divisible by heads.")
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class ContrastiveLoss(nn.Module):
    """Supervised contrastive-style loss, safe for batches without positive pairs."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        diagonal = torch.eye(
            features.size(0), device=features.device, dtype=features.dtype
        )
        positive_mask = positive_mask * (1.0 - diagonal)
        logits = similarity - similarity.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(logits) * (1.0 - diagonal)
        log_prob = logits - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not torch.any(valid):
            return features.sum() * 0.0
        per_sample = -(positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        return per_sample[valid].mean()


class CF2_EfficientNetV2(nn.Module):
    """EfficientNetV2-S + frequency enhancement + attention + gated fusion + Transformer."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = (
            models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        try:
            backbone_model = models.efficientnet_v2_s(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Could not load pretrained EfficientNetV2-S weights. "
                    "Check internet/cache access, or set pretrained=False."
                ) from exc
            raise

        self.backbone = backbone_model.features
        feature_dim = 1280
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.freq = FrequencyEnhancer(feature_dim)
        self.attn = AttentionBlock(feature_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        self.transformer = TransformerBlock(dim=feature_dim, heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feature_map = self.backbone(x)
        spatial = self.pool(feature_map).flatten(1)
        frequency = self.pool(self.freq(feature_map)).flatten(1)
        attended = self.pool(self.attn(feature_map)).flatten(1)
        fusion = torch.cat((spatial, frequency, attended), dim=1)
        gate = self.fusion_gate(fusion)
        fused = gate * spatial + (1.0 - gate) * (
            0.5 * frequency + 0.5 * attended
        )
        fused = self.transformer(fused.unsqueeze(1)).squeeze(1)
        logits = self.classifier(fused)
        return logits, fused


# --------------------------- Training and feature extraction ---------------------------

def train_model(model, loader, output_dir, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    classification_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss()
    history = {"epoch": [], "train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(images)
            loss = classification_loss(logits, labels) + 0.3 * contrastive_loss(
                features, labels
            )
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        epoch_loss = total_loss / max(total, 1)
        epoch_accuracy = correct / max(total, 1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss={epoch_loss:.4f} | train_acc={epoch_accuracy:.4f}"
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir["plots"] / "training_history.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir["plots"] / "epoch_vs_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_accuracy"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir["plots"] / "epoch_vs_accuracy.png", dpi=300)
    plt.close()

    return model, history_df


def extract_features(model, loader):
    model.eval()
    features_list, labels_list, paths = [], [], []
    with torch.no_grad():
        for images, labels, batch_paths in loader:
            images = images.to(DEVICE, non_blocking=True)
            _, features = model(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            paths.extend(batch_paths)

    if not features_list:
        raise RuntimeError("Feature extraction received an empty DataLoader.")
    return (
        np.concatenate(features_list, axis=0),
        np.concatenate(labels_list, axis=0),
        paths
    )


def save_pickle(path, value):
    with open(path, "wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------- Classifiers and evaluation ---------------------------

def build_classifiers():
    # Scaling is fitted only on the training features within each fit.
    return {
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, C=10, random_state=SEED)
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=SEED)
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=SEED),
        "Naive Bayes": GaussianNB()
    }


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)


def train_softmax(X_train, y_train, X_eval, epochs=100):
    model = SoftmaxClassifier(X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=DEVICE)
    x_eval_t = torch.as_tensor(X_eval, dtype=torch.float32, device=DEVICE)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x_eval_t), dim=1)[:, 1].cpu().numpy()
    return (probabilities >= 0.5).astype(np.int64), probabilities, model


def calculate_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "Average Precision": average_precision_score(y_true, y_score)
        if len(np.unique(y_true)) == 2 else np.nan
    }


def evaluate_classifiers(X_train, y_train, X_test, y_test, output_dir):
    results = {}
    metric_rows = []
    classifier_objects = {}

    pred_dir = output_dir["predictions"]
    for name, classifier in build_classifiers().items():
        print(f"Evaluating {name} ...")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        if hasattr(classifier, "predict_proba"):
            scores = classifier.predict_proba(X_test)[:, 1]
        else:
            raw = classifier.decision_function(X_test)
            scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

        results[name] = (predictions, scores)
        classifier_objects[name] = classifier
        row = {"Model": name, **calculate_metrics(y_test, predictions, scores)}
        metric_rows.append(row)
        pd.DataFrame({
            "Ground Truth": y_test,
            "Prediction": predictions,
            "Positive Class Probability/Score": scores
        }).to_csv(pred_dir / f"{name.replace(' ', '_')}_predictions.csv", index=False)

    # Softmax classifier
    print("Evaluating Softmax ...")
    soft_pred, soft_score, softmax_model = train_softmax(
        X_train, y_train, X_test
    )
    results["Softmax"] = (soft_pred, soft_score)
    classifier_objects["Softmax"] = softmax_model
    metric_rows.append({
        "Model": "Softmax",
        **calculate_metrics(y_test, soft_pred, soft_score)
    })
    pd.DataFrame({
        "Ground Truth": y_test,
        "Prediction": soft_pred,
        "Positive Class Probability/Score": soft_score
    }).to_csv(pred_dir / "Softmax_predictions.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir["metrics"] / "All_Model_Metrics.csv", index=False)
    metrics_df.to_csv(output_dir["results"] / "final_classification_results.csv", index=False)

    save_pickle(output_dir["pickle"] / "classifier_predictions.pkl", results)
    for name, classifier in classifier_objects.items():
        save_pickle(
            output_dir["trained_models"] / f"{name.replace(' ', '_')}.pkl",
            classifier
        )

    print("\nTest-set metrics:")
    print(metrics_df.to_string(index=False))
    return results, metrics_df


# --------------------------- Plots ---------------------------

def save_evaluation_plots(results, y_true, output_dir, dataset_name):
    roc_rows, pr_rows = [], []

    for name, (predictions, scores) in results.items():
        safe_name = name.replace(" ", "_")
        cm = confusion_matrix(y_true, predictions, labels=[0, 1])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cbar=False, square=True,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 12}
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"{dataset_name} — {name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir["confusion"] / f"{safe_name}_confusion_matrix.png", dpi=300)
        plt.close()
        pd.DataFrame(
            cm, index=["True Normal", "True Violence"],
            columns=["Pred Normal", "Pred Violence"]
        ).to_csv(output_dir["confusion"] / f"{safe_name}_confusion_matrix.csv")

        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_value = roc_auc_score(y_true, scores)
            roc_rows.append({"Model": name, "ROC-AUC": auc_value})

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_value:.4f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{dataset_name} — {name} ROC")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir["roc"] / f"{safe_name}_ROC.png", dpi=300)
            plt.close()

            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            pr_rows.append({"Model": name, "Average Precision": ap})
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, linewidth=2, label=f"AP={ap:.4f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{dataset_name} — {name} Precision-Recall")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir["pr_curve"] / f"{safe_name}_PR.png", dpi=300)
            plt.close()

    pd.DataFrame(roc_rows).to_csv(output_dir["roc"] / "ROC_AUC_Summary.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(
        output_dir["pr_curve"] / "Average_Precision_Summary.csv", index=False
    )

    # Combined curves
    if len(np.unique(y_true)) == 2:
        plt.figure(figsize=(8, 7))
        for name, (_, scores) in results.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_value = roc_auc_score(y_true, scores)
            plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_value:.4f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset_name} ROC Curve Comparison")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir["roc"] / "All_Models_ROC.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 7))
        for name, (_, scores) in results.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{dataset_name} Precision-Recall Curve Comparison")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir["pr_curve"] / "All_Models_PR_Curves.png", dpi=300)
        plt.close()


# --------------------------- Five-fold CV ---------------------------

def cross_validation_evaluation(X, y, output_dir):
    class_counts = np.bincount(y.astype(int), minlength=2)
    min_class_count = int(class_counts.min())
    if min_class_count < CV_SPLITS:
        warnings.warn(
            f"Skipping {CV_SPLITS}-fold CV: smallest class has {min_class_count} "
            "training feature(s)."
        )
        return pd.DataFrame()

    classifiers = build_classifiers()
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for model_name, classifier in classifiers.items():
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            clf = copy.deepcopy(classifier)
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            clf.fit(X_fold_train, y_fold_train)
            pred = clf.predict(X_fold_val)
            prob = clf.predict_proba(X_fold_val)[:, 1]
            metrics = calculate_metrics(y_fold_val, pred, prob)
            fold_scores.append(metrics)
            print(
                f"CV {model_name} | fold {fold} | "
                f"accuracy={metrics['Accuracy']:.4f} | "
                f"AUC={metrics['ROC-AUC']:.4f}"
            )

        row = {"Classifier": model_name}
        for metric in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"):
            values = [fold[metric] for fold in fold_scores]
            row[f"{metric} Mean"] = float(np.mean(values))
            row[f"{metric} Std"] = float(np.std(values, ddof=0))
        rows.append(row)

    # Softmax CV, trained independently within each fold
    softmax_folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pred, prob, _ = train_softmax(X[train_idx], y[train_idx], X[val_idx], epochs=150)
        metrics = calculate_metrics(y[val_idx], pred, prob)
        softmax_folds.append(metrics)
        print(
            f"CV Softmax | fold {fold} | accuracy={metrics['Accuracy']:.4f} | "
            f"AUC={metrics['ROC-AUC']:.4f}"
        )
    soft_row = {"Classifier": "Softmax"}
    for metric in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"):
        values = [fold[metric] for fold in softmax_folds]
        soft_row[f"{metric} Mean"] = float(np.mean(values))
        soft_row[f"{metric} Std"] = float(np.std(values, ddof=0))
    rows.append(soft_row)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(
        output_dir["cross_validation"] / "FiveFold_CrossValidation.csv",
        index=False
    )
    publication = cv_df[["Classifier"]].copy()
    for metric in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"):
        publication[metric] = [
            f"{mean * 100:.2f} ± {std * 100:.2f}"
            for mean, std in zip(cv_df[f"{metric} Mean"], cv_df[f"{metric} Std"])
        ]
    publication.to_csv(
        output_dir["cross_validation"] / "Publication_Table.csv",
        index=False
    )
    return cv_df


# --------------------------- One dataset run; reused for HF and VF ---------------------------

def run_dataset(dataset_name):
    dataset_dir = DATA_ROOT / dataset_name
    train_dir = dataset_dir / "train_data"
    test_dir = dataset_dir / "test_data"
    output_base = RESULT_ROOT / f"{dataset_name.lower()}_result"
    output_base.mkdir(parents=True, exist_ok=True)
    output_dirs = make_output_dirs(output_base)

    print("\n" + "=" * 80)
    print(f"Starting dataset: {dataset_name}")
    print(f"Train directory: {train_dir}")
    print(f"Test directory : {test_dir}")
    print(f"Output directory: {output_base}")
    print("=" * 80)

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset_name} requires both '{train_dir}' and '{test_dir}'. "
            "Check the expected folder structure at the top of this script."
        )

    train_samples = collect_samples(train_dir)
    test_samples = collect_samples(test_dir)
    validate_split(train_samples, dataset_name, "train")
    validate_split(test_samples, dataset_name, "test")

    train_loader = DataLoader(
        ViolenceImageDataset(train_samples),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    train_eval_loader = DataLoader(
        ViolenceImageDataset(train_samples),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        ViolenceImageDataset(test_samples),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )

    model = CF2_EfficientNetV2(pretrained=True)
    model, history_df = train_model(model, train_loader, output_dirs)

    # Save the trained neural model and training history.
    torch.save(
        {
            "dataset": dataset_name,
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "image_size": IMAGE_SIZE,
            "seed": SEED
        },
        output_dirs["trained_models"] / "cf2_efficientnetv2_s.pth"
    )

    X_train, y_train, train_paths = extract_features(model, train_eval_loader)
    X_test, y_test, test_paths = extract_features(model, test_loader)

    print(f"{dataset_name} feature shapes: train={X_train.shape}, test={X_test.shape}")
    for name, value in (
        ("X_train", X_train), ("y_train", y_train),
        ("X_test", X_test), ("y_test", y_test),
        ("train_paths", train_paths), ("test_paths", test_paths)
    ):
        save_pickle(output_dirs["features"] / f"{dataset_name}_{name}.pkl", value)

    # Classifiers use the same Transformer-refined feature vectors.
    predictions, metrics_df = evaluate_classifiers(
        X_train, y_train, X_test, y_test, output_dirs
    )
    save_evaluation_plots(predictions, y_test, output_dirs, dataset_name)
    cv_df = cross_validation_evaluation(X_train, y_train, output_dirs)

    metrics_df.to_csv(output_dirs["results"] / "final_metrics.csv", index=False)
    with open(output_dirs["pickle"] / "run_summary.pkl", "wb") as file:
        pickle.dump(
            {
                "dataset": dataset_name,
                "test_metrics": metrics_df,
                "cross_validation": cv_df,
                "training_history": history_df,
                "train_sample_count": len(train_samples),
                "test_sample_count": len(test_samples)
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(f"Completed {dataset_name}. Results saved to: {output_base.resolve()}")
    return metrics_df, cv_df


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for dataset_name in DATASET_NAMES:
        set_seed(SEED)
        metrics_df, cv_df = run_dataset(dataset_name)
        summaries[dataset_name] = {
            "test_metrics": metrics_df,
            "cross_validation": cv_df
        }

    # Convenient top-level summary while keeping all individual outputs separate.
    summary_rows = []
    for dataset_name, result in summaries.items():
        for _, row in result["test_metrics"].iterrows():
            summary_rows.append({"Dataset": dataset_name, **row.to_dict()})
    pd.DataFrame(summary_rows).to_csv(
        RESULT_ROOT / "HF_VF_combined_test_metrics.csv", index=False
    )
    print("\nBoth datasets completed successfully.")
    print(f"Separate results: {(RESULT_ROOT / 'hf_result').resolve()} and "
          f"{(RESULT_ROOT / 'vf_result').resolve()}")


if __name__ == "__main__":
    main()
