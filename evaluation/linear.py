#!/usr/bin/env python3
#
# linear.py  Andrew Belles  April 13th, 2026
#
# Linear probe evaluation over compression parquet embeddings.
#

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from compression.train_utils import load_config

SPLITS = ("training", "validation", "test")


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(col for col in frame.columns if col.startswith("embedding_"))
    if not columns:
        raise ValueError("frame has no embedding columns")
    return columns


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "linear.yaml"
DEFAULT_CONFIG = {
    "seed": 7,
    "classifier": "logistic",
    "optuna": {
        "trials": 20,
        "target_metric": "f1_macro",
    },
    "c_min": 1e-4,
    "c_max": 1.0,
    "max_iter": 10000,
    "tol": 1e-3,
    "device": "cuda",
    "torch_epochs": 200,
    "torch_lr": 0.05,
    "torch_batch_size": 2048,
}


def log(message: str) -> None:
    print(message, flush=True)


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe evaluation over representation parquets.")
    parser.add_argument(
        "-p",
        "--parquet",
        type=Path,
        required=True,
        help="Path to a pipeline parquet (e.g. representation/data/barlow_fma_small_mel.parquet).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {}
    for split in SPLITS:
        sub = df[df["split"] == split].copy()
        if sub.empty:
            raise ValueError(f"parquet has no rows for split={split}")
        frames[split] = sub
    return frames




def build_features(
    frames: dict[str, pd.DataFrame],
    embedding_columns: list[str],
) -> dict[str, np.ndarray]:
    return {
        split: frame[embedding_columns].to_numpy(dtype=np.float32, copy=True)
        for split, frame in frames.items()
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class TorchLogisticProbe:
    def __init__(
        self,
        c_value: float,
        device: torch.device,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        seed: int,
    ):
        self.c_value = float(c_value)
        self.device = device
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.classes_ = np.unique(labels)
        n_classes = int(len(self.classes_))
        n_features = int(features.shape[1])

        class_to_index = {int(label): index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([class_to_index[int(label)] for label in labels], dtype=np.int64)

        mean_np = features.mean(axis=0, keepdims=True).astype("float32")
        std_np = np.maximum(features.std(axis=0, keepdims=True), 1e-6).astype("float32")
        self.mean = torch.from_numpy(mean_np).to(self.device)
        self.std = torch.from_numpy(std_np).to(self.device)

        generator = torch.Generator().manual_seed(self.seed)
        weight = torch.empty((n_classes, n_features), device=self.device)
        torch.nn.init.xavier_uniform_(weight)
        bias = torch.zeros(n_classes, device=self.device)
        weight.requires_grad_(True)
        bias.requires_grad_(True)

        optimizer = torch.optim.Adam([weight, bias], lr=self.learning_rate)
        labels_tensor = torch.from_numpy(encoded)
        n_samples = int(features.shape[0])
        l2_scale = 0.5 / max(self.c_value * n_samples, 1e-8)

        for _ in range(self.epochs):
            permutation = torch.randperm(n_samples, generator=generator)
            for start in range(0, n_samples, self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_x = torch.from_numpy(features[indices.numpy()]).to(self.device)
                batch_y = labels_tensor[indices].to(self.device)
                batch_x = (batch_x - self.mean) / self.std

                logits = F.linear(batch_x, weight, bias)
                loss = F.cross_entropy(logits, batch_y) + l2_scale * weight.pow(2).sum()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.weight = weight.detach()
        self.bias = bias.detach()
        return self

    @torch.no_grad()
    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None or self.weight is None or self.bias is None:
            raise RuntimeError("TorchLogisticProbe must be fit before inference")

        batches: list[np.ndarray] = []
        for start in range(0, features.shape[0], self.batch_size):
            batch_x = torch.from_numpy(features[start : start + self.batch_size]).to(self.device)
            batch_x = (batch_x - self.mean) / self.std
            logits = F.linear(batch_x, self.weight, self.bias)
            batches.append(logits.cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("TorchLogisticProbe must be fit before prediction")
        logits = self.decision_function(features)
        return self.classes_[np.argmax(logits, axis=1)]

    def weight_matrix(self) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("TorchLogisticProbe must be fit before reading weights")
        return self.weight.detach().cpu().numpy()


def build_estimator(classifier: str, c_value: float, max_iter: int, tol: float, config: dict):
    if classifier == "logistic":
        return TorchLogisticProbe(
            c_value=c_value,
            device=resolve_device(str(config["device"])),
            epochs=int(config["torch_epochs"]),
            learning_rate=float(config["torch_lr"]),
            batch_size=int(config["torch_batch_size"]),
            seed=int(config["seed"]),
        )

    if classifier == "knn":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=max(1, int(round(c_value))))),
            ]
        )

    if classifier == "sparse_logistic":
        base = LogisticRegression(
            penalty="l1",
            C=c_value,
            solver="liblinear",
            max_iter=max_iter,
            tol=tol,
        )
    elif classifier == "svm":
        base = LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual="auto",
            C=c_value,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        raise ValueError(f"unsupported classifier: {classifier}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", OneVsRestClassifier(base)),
        ]
    )



def get_decision_scores(estimator, features: np.ndarray, n_classes: int) -> np.ndarray:
    if isinstance(estimator, TorchLogisticProbe):
        scores = estimator.decision_function(features)
        if scores.shape[1] != n_classes:
            full_scores = np.full((scores.shape[0], n_classes), float(np.min(scores) - 1.0), dtype=np.float64)
            class_ids = np.asarray(estimator.classes_, dtype=int)
            full_scores[:, class_ids] = scores
            return full_scores
        return scores

    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    else:
        scores = estimator.decision_function(features)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)

    classifier = estimator.named_steps["classifier"]
    class_ids = np.asarray(classifier.classes_, dtype=int)
    full_scores = np.full((scores.shape[0], n_classes), float(np.min(scores) - 1.0), dtype=np.float64)
    if scores.shape[1] == len(class_ids):
        full_scores[:, class_ids] = scores
    elif len(class_ids) == 1 and scores.shape[1] == 2:
        full_scores[:, class_ids[0]] = scores[:, 1]
    else:
        raise ValueError(
            f"could not align decision scores: scores shape={scores.shape}, class_ids shape={class_ids.shape}"
        )
    return full_scores



def compute_pr_auc_macro(scores: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    binarized = label_binarize(labels, classes=np.arange(n_classes))
    present_mask = np.any(binarized == 1, axis=0)
    if not np.any(present_mask):
        return 0.0
    return float(average_precision_score(binarized[:, present_mask], scores[:, present_mask], average="macro"))


def compute_metrics(
    estimator,
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    predictions = estimator.predict(features)
    scores = get_decision_scores(estimator, features, n_classes=n_classes)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "pr_auc_macro": compute_pr_auc_macro(scores, labels, n_classes=n_classes),
    }


def optimize_hyperparameters(
    classifier: str,
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, float], float]:
    max_iter = int(config["max_iter"])
    tol = float(config["tol"])
    trials = int(config["optuna"]["trials"])
    seed = int(config["seed"])
    target_metric = str(config["optuna"]["target_metric"])
    n_classes = len(np.unique(np.concatenate([y_train, y_val])))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        if classifier == "knn":
            choices = [int(value) for value in config.get("knn_neighbors", [3, 5, 9, 15, 25])]
            c_value = float(trial.suggest_categorical("n_neighbors", choices))
        else:
            c_value = trial.suggest_float("C", float(config["c_min"]), float(config["c_max"]), log=True)
        estimator = build_estimator(classifier, c_value, max_iter=max_iter, tol=tol, config=config)
        estimator.fit(x_train, y_train)
        metrics = compute_metrics(estimator, x_val, y_val, n_classes=n_classes)
        return float(metrics[target_metric])

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    if classifier == "knn":
        return {"C": float(study.best_params["n_neighbors"]), "n_neighbors": int(study.best_params["n_neighbors"])}, float(study.best_value)
    return study.best_params, float(study.best_value)


def probe_group(
    group: pd.DataFrame,
    classifier: str,
    config: dict,
) -> dict:
    cols = embedding_columns(group)
    frames = split_frames(group)
    feats = build_features(frames, cols)

    encoder = LabelEncoder()
    encoder.fit(pd.concat([f["genre_top"] for f in frames.values()], ignore_index=True))
    labels = {s: encoder.transform(f["genre_top"]) for s, f in frames.items()}
    n_classes = len(encoder.classes_)

    best_params, _ = optimize_hyperparameters(
        classifier, config,
        feats["training"], labels["training"],
        feats["validation"], labels["validation"],
    )
    estimator = build_estimator(
        classifier, float(best_params["C"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        config=config,
    )
    estimator.fit(feats["training"], labels["training"])
    val_m = compute_metrics(estimator, feats["validation"], labels["validation"], n_classes=n_classes)
    test_m = compute_metrics(estimator, feats["test"], labels["test"], n_classes=n_classes)

    first = group.iloc[0]
    return {
        "method": str(first.get("method", "")),
        "family": str(first.get("family", "")),
        "dataset": str(first.get("dataset", "")),
        "ratio_percent": None if pd.isna(first.get("ratio_percent", float("nan"))) else int(first["ratio_percent"]),
        "augmentation": str(first.get("augmentation", "")),
        "sensing_pair": str(first.get("sensing_pair", "")),
        "seed": int(first.get("seed", 0)),
        "best_c": float(best_params["C"]),
        "validation_f1_macro": val_m["f1_macro"],
        "validation_pr_auc_macro": val_m["pr_auc_macro"],
        "validation_accuracy": val_m["accuracy"],
        "test_f1_macro": test_m["f1_macro"],
        "test_pr_auc_macro": test_m["pr_auc_macro"],
        "test_accuracy": test_m["accuracy"],
        "n_train": int(len(labels["training"])),
        "n_val": int(len(labels["validation"])),
        "n_test": int(len(labels["test"])),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parquet_path = args.parquet.expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    classifier = str(config["classifier"])
    df = pd.read_parquet(parquet_path)
    report(f"START module=evaluation.linear parquet={parquet_path} classifier={classifier} rows={len(df)}")

    group_cols = ["method"]
    if "ratio_percent" in df.columns:
        group_cols.append("ratio_percent")

    records = []
    for keys, group in df.groupby(group_cols, dropna=False):
        label = keys if isinstance(keys, str) else "_".join(str(k) for k in keys)
        log(f"probing group={label} n={len(group)}")
        records.append(probe_group(group, classifier, config))

    summary = pd.DataFrame.from_records(records)
    data_root = Path(__file__).resolve().parent / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    out_path = data_root / f"{parquet_path.stem}_{classifier}_summary.csv"
    summary.to_csv(out_path, index=False)

    log(summary[["method", "ratio_percent", "validation_f1_macro", "test_f1_macro", "test_pr_auc_macro"]].to_string(index=False))
    report(f"DONE module=evaluation.linear saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
