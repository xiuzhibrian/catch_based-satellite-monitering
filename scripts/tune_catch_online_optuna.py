# -*- coding: utf-8 -*-
"""
Optuna automated hyperparameter tuning wrapper for the user's CATCH online runner.

Design:
1) Train only on historical normal data (--train-data).
2) Select hyperparameters only on a labeled validation stream (--val-input).
3) Optionally evaluate the final best configuration once on a held-out test stream (--test-input).
4) Do not modify the original CATCH network or online runner. This script calls:
       scripts/run_catch_online_stride64.py
   as a subprocess and reads its point-level output.

Recommended placement:
    CATCH/
    ├─ scripts/
    │  ├─ run_catch_online_stride64.py
    │  ├─ catch_online_evaluation.py
    │  └─ tune_catch_online_optuna.py   <-- this file
    └─ ...

Example:
python ./scripts/tune_catch_online_optuna.py \
  --train-data ./dataset/dataset/anomaly_detect/running_data_normal1_CATCH.csv \
  --val-input ./dataset/dataset/anomaly_detect/validation_labeled.csv \
  --test-input ./dataset/dataset/anomaly_detect/test_labeled.csv \
  --n-trials 20 \
  --device cuda

The script performs staged tuning by default:
geometry -> patch -> loss -> optim -> model -> regularization -> threshold

Each stage starts from the best parameters found by the previous stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "\n[ERROR] Optuna is not installed.\n"
        "Install it in the SAME Python environment used to run CATCH:\n\n"
        "    python -m pip install optuna\n\n"
        "Then verify:\n\n"
        "    python -c \"import optuna; print(optuna.__version__)\"\n"
    ) from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )
except ImportError as exc:
    raise SystemExit(
        "\n[ERROR] scikit-learn is required for validation metrics.\n"
        "Install it with:\n\n"
        "    python -m pip install scikit-learn\n"
    ) from exc


# ---------------------------------------------------------------------
# Baseline parameters
# ---------------------------------------------------------------------
# Baseline copied from the current adjustable-parameter runner example.
BASE_PARAMS: Dict[str, object] = {
    "lr": 5e-4,
    "Mlr": 1e-5,
    "auxi_lambda": 0.05,
    "batch_size": 32,
    "cf_dim": 32,
    "d_ff": 256,
    "d_model": 256,
    "dc_lambda": 0.1,
    "dropout": 0.05,
    "e_layers": 3,
    "head_dim": 32,
    "head_dropout": 0.1,
    "n_heads": 1,
    "num_epochs": 5,
    "patch_size": 16,
    "patch_stride": 8,
    "score_lambda": 0.05,
    "seq_len": 192,
    "online_stride": 64,
    "inference_patch_size": 32,
    "inference_patch_stride": 1,
    "anomaly_ratio": 14.0,
}

DEFAULT_STAGES = [
    "geometry",
    "patch",
    "loss",
    "optim",
    "model",
    "regularization",
    "threshold",
]

ALLOWED_STAGES = set(DEFAULT_STAGES + ["all"])


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return data


def tail_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def finite_or_none(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def validate_online_geometry(params: dict) -> None:
    seq_len = int(params["seq_len"])
    online_stride = int(params["online_stride"])
    ips = int(params["inference_patch_size"])
    ipstride = int(params["inference_patch_stride"])

    if seq_len <= 0:
        raise ValueError("seq_len must be > 0.")
    if online_stride <= 0:
        raise ValueError("online_stride must be > 0.")
    if ips <= 0:
        raise ValueError("inference_patch_size must be > 0.")
    if ipstride <= 0:
        raise ValueError("inference_patch_stride must be > 0.")

    if online_stride > seq_len:
        raise ValueError("online_stride must be <= seq_len.")
    if ips > seq_len:
        raise ValueError("inference_patch_size must be <= seq_len.")
    if ipstride > ips:
        raise ValueError(
            "inference_patch_stride must be <= inference_patch_size."
        )

    # Required by the current formal online score-release rule.
    required = online_stride + 2 * (ips - 1)
    if seq_len < required:
        raise ValueError(
            "Invalid online geometry: "
            f"seq_len={seq_len} < online_stride + 2*(inference_patch_size-1) "
            f"= {required}."
        )

    patch_size = int(params["patch_size"])
    patch_stride = int(params["patch_stride"])
    if patch_size <= 0 or patch_size > seq_len:
        raise ValueError("patch_size must be in [1, seq_len].")
    if patch_stride <= 0 or patch_stride > patch_size:
        raise ValueError("patch_stride must be in [1, patch_size].")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def _binary_arrays(df: pd.DataFrame):
    required = {"ground_truth_label", "pred_label", "online_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Runner output is missing required labeled-validation columns: "
            + ", ".join(sorted(missing))
            + ". Validation/Test data must contain ground-truth labels."
        )

    mask = (
        df["ground_truth_label"].notna()
        & df["pred_label"].notna()
        & df["online_score"].notna()
    )
    scored = df.loc[mask].copy()
    if scored.empty:
        raise ValueError(
            "No scored rows with ground-truth labels were produced."
        )

    y_true = (
        pd.to_numeric(scored["ground_truth_label"], errors="raise")
        .to_numpy()
        != 0
    ).astype(np.int64)

    y_pred = (
        pd.to_numeric(scored["pred_label"], errors="raise")
        .to_numpy()
        != 0
    ).astype(np.int64)

    scores = pd.to_numeric(
        scored["online_score"], errors="raise"
    ).to_numpy(dtype=np.float64)

    return scored, y_true, y_pred, scores


def event_metrics(df: pd.DataFrame) -> dict:
    """
    Simple strict event metrics on the original sample timeline.

    An event is a contiguous run of ground_truth_label == 1.
    It is counted as detected if any scored pred_label == 1 occurs inside it.
    """
    if "ground_truth_label" not in df.columns:
        return {
            "event_count": 0,
            "event_detected": 0,
            "event_recall": float("nan"),
            "event_first_alarm_delay_mean_samples": float("nan"),
        }

    gt = pd.to_numeric(
        df["ground_truth_label"], errors="coerce"
    ).fillna(0).to_numpy()
    gt = (gt != 0).astype(np.int64)

    pred = pd.to_numeric(
        df.get("pred_label", pd.Series([np.nan] * len(df))),
        errors="coerce",
    ).to_numpy()

    sample_index = pd.to_numeric(
        df.get(
            "sample_index",
            pd.Series(np.arange(1, len(df) + 1)),
        ),
        errors="coerce",
    ).to_numpy()

    events = []
    start = None
    for i, label in enumerate(gt):
        if label == 1 and start is None:
            start = i
        if start is not None and (label == 0 or i == len(gt) - 1):
            end = i if (label == 1 and i == len(gt) - 1) else i - 1
            events.append((start, end))
            start = None

    if not events:
        return {
            "event_count": 0,
            "event_detected": 0,
            "event_recall": float("nan"),
            "event_first_alarm_delay_mean_samples": float("nan"),
        }

    detected = 0
    delays = []

    for start, end in events:
        event_pred = pred[start : end + 1]
        alarm_local = np.where(event_pred == 1)[0]
        if len(alarm_local):
            detected += 1
            alarm_row = start + int(alarm_local[0])
            delay = sample_index[alarm_row] - sample_index[start]
            if np.isfinite(delay):
                delays.append(float(delay))

    return {
        "event_count": int(len(events)),
        "event_detected": int(detected),
        "event_recall": float(detected / len(events)),
        "event_first_alarm_delay_mean_samples": (
            float(np.mean(delays)) if delays else float("nan")
        ),
    }


def compute_metrics(output_csv: Path) -> dict:
    df = pd.read_csv(output_csv)
    scored, y_true, y_pred, scores = _binary_arrays(df)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    if len(np.unique(y_true)) >= 2:
        roc_auc = float(roc_auc_score(y_true, scores))
        pr_auc = float(average_precision_score(y_true, scores))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    metrics = {
        "scored_samples": int(len(scored)),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    metrics.update(event_metrics(df))

    # Runtime / release diagnostics if present.
    if "trigger_end_to_end_ms" in scored.columns:
        vals = pd.to_numeric(
            scored["trigger_end_to_end_ms"], errors="coerce"
        ).dropna()
        metrics["trigger_e2e_mean_ms"] = (
            float(vals.mean()) if len(vals) else float("nan")
        )
        metrics["trigger_e2e_p95_ms"] = (
            float(vals.quantile(0.95)) if len(vals) else float("nan")
        )

    if "model_inference_ms" in scored.columns:
        vals = pd.to_numeric(
            scored["model_inference_ms"], errors="coerce"
        ).dropna()
        metrics["model_inference_mean_ms"] = (
            float(vals.mean()) if len(vals) else float("nan")
        )
        metrics["model_inference_p95_ms"] = (
            float(vals.quantile(0.95)) if len(vals) else float("nan")
        )

    if "detection_delay_samples" in scored.columns:
        vals = pd.to_numeric(
            scored["detection_delay_samples"], errors="coerce"
        ).dropna()
        metrics["score_release_delay_mean_samples"] = (
            float(vals.mean()) if len(vals) else float("nan")
        )
        metrics["score_release_delay_p95_samples"] = (
            float(vals.quantile(0.95)) if len(vals) else float("nan")
        )

    return metrics


def objective_value(metrics: dict, name: str) -> float:
    if name in {"f1", "pr_auc", "roc_auc"}:
        value = finite_or_none(metrics.get(name))
        if value is None:
            raise ValueError(
                f"Objective metric '{name}' is unavailable. "
                "Check that Validation contains both normal and anomaly labels."
            )
        return value

    if name == "composite":
        # Bounded detection-oriented objective.
        # Delay is deliberately reported/filtered separately rather than mixed
        # into this scalar because its scale depends on the online geometry.
        f1 = finite_or_none(metrics.get("f1")) or 0.0
        pr_auc = finite_or_none(metrics.get("pr_auc")) or 0.0
        recall = finite_or_none(metrics.get("recall")) or 0.0
        event_recall = finite_or_none(metrics.get("event_recall"))
        event_recall = 0.0 if event_recall is None else event_recall
        fpr = finite_or_none(metrics.get("fpr")) or 0.0
        return (
            0.40 * f1
            + 0.20 * pr_auc
            + 0.20 * recall
            + 0.10 * event_recall
            - 0.10 * fpr
        )

    raise ValueError(f"Unknown objective: {name}")


# ---------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------
def suggest_for_stage(
    trial: optuna.Trial,
    stage: str,
    current: dict,
) -> dict:
    p = {}

    if stage == "geometry":
        p["seq_len"] = trial.suggest_categorical(
            "seq_len", [128, 192]
        )
        p["inference_patch_size"] = trial.suggest_categorical(
            "inference_patch_size", [8, 16, 32, 48]
        )
        p["online_stride"] = trial.suggest_categorical(
            "online_stride", [16, 32, 64]
        )
        p["inference_patch_stride"] = trial.suggest_categorical(
            "inference_patch_stride", [1, 2, 4]
        )

    elif stage == "patch":
        p["patch_size"] = trial.suggest_categorical(
            "patch_size", [8, 16, 32]
        )
        p["patch_stride"] = trial.suggest_categorical(
            "patch_stride", [4, 8, 16]
        )

    elif stage == "loss":
        p["score_lambda"] = trial.suggest_categorical(
            "score_lambda", [0.01, 0.025, 0.05, 0.10, 0.25]
        )
        p["dc_lambda"] = trial.suggest_categorical(
            "dc_lambda", [0.05, 0.10, 0.20]
        )
        p["auxi_lambda"] = trial.suggest_categorical(
            "auxi_lambda", [0.025, 0.05, 0.10]
        )

    elif stage == "optim":
        p["lr"] = trial.suggest_float(
            "lr", 1e-4, 1e-3, log=True
        )
        p["Mlr"] = trial.suggest_float(
            "Mlr", 1e-6, 3e-5, log=True
        )
        p["batch_size"] = trial.suggest_categorical(
            "batch_size", [16, 32, 64]
        )

    elif stage == "model":
        p["d_model"] = trial.suggest_categorical(
            "d_model", [64, 128, 256, 512]
        )
        p["d_ff"] = trial.suggest_categorical(
            "d_ff", [128, 256, 512, 1024]
        )
        p["e_layers"] = trial.suggest_categorical(
            "e_layers", [1, 2, 3]
        )
        p["n_heads"] = trial.suggest_categorical(
            "n_heads", [1, 2, 4]
        )
        p["head_dim"] = trial.suggest_categorical(
            "head_dim", [16, 32, 64]
        )
        p["cf_dim"] = trial.suggest_categorical(
            "cf_dim", [16, 32, 64]
        )

    elif stage == "regularization":
        p["dropout"] = trial.suggest_categorical(
            "dropout", [0.0, 0.05, 0.10, 0.20]
        )
        p["head_dropout"] = trial.suggest_categorical(
            "head_dropout", [0.0, 0.05, 0.10, 0.20]
        )
        p["num_epochs"] = trial.suggest_categorical(
            "num_epochs", [5, 10, 20]
        )
        p["patience"] = trial.suggest_categorical(
            "patience", [3, 5, 10]
        )

    elif stage == "threshold":
        p["anomaly_ratio"] = trial.suggest_categorical(
            "anomaly_ratio", [0.1, 0.5, 1.0, 2.0, 5.0]
        )

    elif stage == "all":
        # Joint search. Use this only when you have enough compute.
        for substage in [
            "geometry",
            "patch",
            "loss",
            "optim",
            "model",
            "regularization",
            "threshold",
        ]:
            p.update(suggest_for_stage(trial, substage, current))

    else:
        raise ValueError(f"Unsupported tuning stage: {stage}")

    merged = dict(current)
    merged.update(p)

    # Hard constraints: prune illegal combinations before starting CATCH.
    try:
        validate_online_geometry(merged)
    except ValueError as exc:
        raise optuna.TrialPruned(str(exc)) from exc

    # Lightweight model consistency rule.
    if int(merged["d_model"]) % int(merged["n_heads"]) != 0:
        raise optuna.TrialPruned(
            "d_model must be divisible by n_heads."
        )

    return p


# ---------------------------------------------------------------------
# Runner invocation
# ---------------------------------------------------------------------
def build_runner_command(
    args,
    params: dict,
    monitor_input: Path,
    bundle: Path,
    output_csv: Path,
    *,
    train: bool,
) -> list[str]:
    cmd = [
        args.python_executable,
        str(args.runner),
        "--input",
        str(monitor_input),
        "--bundle",
        str(bundle),
        "--output",
        str(output_csv),
        "--device",
        args.device,
    ]

    if train:
        cmd += [
            "--train-data",
            str(args.train_data),
            "--model-hyper-params",
            json.dumps(params, ensure_ascii=False),
            "--calibration-ratio",
            str(args.calibration_ratio),
            "--seed",
            str(args.seed),
        ]

    if args.sample_period is not None:
        cmd += ["--sample-period", str(args.sample_period)]

    return cmd


def run_runner(
    args,
    params: dict,
    monitor_input: Path,
    bundle: Path,
    output_csv: Path,
    log_path: Path,
    *,
    train: bool,
) -> None:
    bundle.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_runner_command(
        args,
        params=params,
        monitor_input=monitor_input,
        bundle=bundle,
        output_csv=output_csv,
        train=train,
    )

    with log_path.open("w", encoding="utf-8") as log:
        log.write("[COMMAND]\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"CATCH runner failed with exit code {proc.returncode}.\n"
            f"Log: {log_path}\n"
            f"Last log lines:\n{tail_text(log_path)}"
        )

    if not output_csv.exists():
        raise RuntimeError(
            f"CATCH runner returned success but did not create: {output_csv}"
        )


# ---------------------------------------------------------------------
# Optuna stage
# ---------------------------------------------------------------------
def run_stage(args, stage: str, current_params: dict) -> tuple[dict, dict]:
    stage_dir = args.output_dir / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.output_dir / "optuna_studies.db"
    storage = f"sqlite:///{db_path.resolve().as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=f"{args.study_prefix}_{stage}",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    print("\n" + "=" * 88)
    print(f"[STAGE] {stage}")
    print(f"[TRIALS TO ADD] {args.n_trials}")
    print(f"[OBJECTIVE] {args.objective}")
    print("=" * 88)

    def objective(trial: optuna.Trial) -> float:
        suggested = suggest_for_stage(
            trial,
            stage=stage,
            current=current_params,
        )

        params = dict(current_params)
        params.update(suggested)

        trial_tag = f"trial_{trial.number:05d}"
        bundle = stage_dir / f"{trial_tag}.pt"
        output_csv = stage_dir / f"{trial_tag}_points.csv"
        log_path = stage_dir / f"{trial_tag}.log"

        trial.set_user_attr("full_params", params)

        try:
            run_runner(
                args,
                params=params,
                monitor_input=args.val_input,
                bundle=bundle,
                output_csv=output_csv,
                log_path=log_path,
                train=True,
            )
            metrics = compute_metrics(output_csv)
            value = objective_value(metrics, args.objective)

            for key, metric_value in metrics.items():
                v = finite_or_none(metric_value)
                if v is not None:
                    trial.set_user_attr(key, v)

            print(
                f"[{stage}] trial={trial.number} "
                f"value={value:.6f} "
                f"F1={metrics['f1']:.6f} "
                f"Recall={metrics['recall']:.6f} "
                f"PR-AUC={metrics['pr_auc']:.6f} "
                f"FPR={metrics['fpr']:.6f}"
            )

            return float(value)

        except (RuntimeError, ValueError) as exc:
            trial.set_user_attr("error", str(exc)[-4000:])
            print(
                f"[{stage}] trial={trial.number} pruned: "
                f"{str(exc).splitlines()[0]}"
            )
            raise optuna.TrialPruned(str(exc)) from exc

        finally:
            if not args.keep_trial_files:
                # Tuning can generate many GB of bundles.
                for p in (bundle, output_csv):
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError:
                        pass

                # Keep logs only for failed/pruned trials. Successful logs are
                # not needed once trial metrics are stored in SQLite/CSV.
                # We cannot reliably know state here until objective returns,
                # so successful log cleanup is handled after optimize.

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # Remove logs from COMPLETE trials unless user asks to keep everything.
    if not args.keep_trial_files:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                log = stage_dir / f"trial_{t.number:05d}.log"
                try:
                    if log.exists():
                        log.unlink()
                except OSError:
                    pass

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError(
            f"Stage '{stage}' has no successful trials. "
            f"Check logs under {stage_dir}."
        )

    best_trial = study.best_trial
    best_stage_params = dict(best_trial.params)
    updated = dict(current_params)
    updated.update(best_stage_params)

    stage_summary = {
        "stage": stage,
        "objective": args.objective,
        "best_value": float(best_trial.value),
        "best_stage_params": best_stage_params,
        "best_full_params": updated,
        "best_trial_number": int(best_trial.number),
        "best_metrics": {
            key: value
            for key, value in best_trial.user_attrs.items()
            if key not in {"full_params", "error"}
        },
    }

    save_json(
        stage_dir / "best_stage_result.json",
        stage_summary,
    )
    study.trials_dataframe().to_csv(
        stage_dir / "trials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[BEST {stage}] value={best_trial.value:.6f}")
    print(
        "[BEST PARAMS] "
        + json.dumps(best_stage_params, ensure_ascii=False)
    )

    return updated, stage_summary


# ---------------------------------------------------------------------
# Final validation / test
# ---------------------------------------------------------------------
def run_final(args, best_params: dict) -> dict:
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    bundle = final_dir / "best_model_bundle.pt"
    val_output = final_dir / "best_validation_points.csv"
    val_log = final_dir / "best_validation.log"

    print("\n" + "=" * 88)
    print("[FINAL] Retrain once with the final best parameters")
    print("[FINAL] Evaluate on Validation and save the final bundle")
    print("=" * 88)

    run_runner(
        args,
        params=best_params,
        monitor_input=args.val_input,
        bundle=bundle,
        output_csv=val_output,
        log_path=val_log,
        train=True,
    )
    val_metrics = compute_metrics(val_output)
    save_json(final_dir / "best_validation_metrics.json", val_metrics)

    result = {
        "best_params": best_params,
        "validation_metrics": val_metrics,
        "bundle": str(bundle),
        "validation_output": str(val_output),
    }

    if args.test_input is not None:
        test_output = final_dir / "final_test_points.csv"
        test_log = final_dir / "final_test.log"

        print("\n" + "=" * 88)
        print("[TEST] Held-out test is evaluated ONCE using the saved best bundle")
        print("=" * 88)

        # IMPORTANT: do not pass --train-data here.
        run_runner(
            args,
            params=best_params,
            monitor_input=args.test_input,
            bundle=bundle,
            output_csv=test_output,
            log_path=test_log,
            train=False,
        )

        test_metrics = compute_metrics(test_output)
        save_json(final_dir / "final_test_metrics.json", test_metrics)

        result["test_metrics"] = test_metrics
        result["test_output"] = str(test_output)

    save_json(final_dir / "final_result.json", result)
    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated staged Optuna tuning for CATCH online monitoring."
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Historical NORMAL training data.",
    )
    parser.add_argument(
        "--val-input",
        type=Path,
        required=True,
        help="Labeled validation stream used for hyperparameter selection.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help=(
            "Optional labeled held-out test stream. "
            "It is used only once after tuning finishes."
        ),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("./scripts/run_catch_online_stride64.py"),
        help="Path to the existing adjustable CATCH online runner.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/optuna_catch"),
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help=(
            "Python executable used to launch the CATCH runner. "
            "Default: the same interpreter running this Optuna script."
        ),
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of new Optuna trials added per stage.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=(
            "Comma-separated stages. Default: "
            + ",".join(DEFAULT_STAGES)
            + ". Use --stages all for one large joint search."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["f1", "pr_auc", "roc_auc", "composite"],
        default="f1",
        help="Validation metric maximized by Optuna.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
    )
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=None,
        help="Sampling period in seconds, e.g. 0.001.",
    )
    parser.add_argument(
        "--study-prefix",
        default="catch_online",
    )
    parser.add_argument(
        "--base-params-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file whose values override BASE_PARAMS "
            "before tuning starts."
        ),
    )
    parser.add_argument(
        "--keep-trial-files",
        action="store_true",
        help=(
            "Keep every trial bundle/output/log. By default, large temporary "
            "trial bundles and outputs are deleted after metrics are recorded."
        ),
    )

    args = parser.parse_args()

    if args.n_trials <= 0:
        parser.error("--n-trials must be > 0.")
    if not (0.0 < args.calibration_ratio < 1.0):
        parser.error("--calibration-ratio must be in (0, 1).")

    stages = [
        s.strip().lower()
        for s in args.stages.split(",")
        if s.strip()
    ]
    if not stages:
        parser.error("--stages is empty.")
    unknown = [s for s in stages if s not in ALLOWED_STAGES]
    if unknown:
        parser.error(
            "Unknown stage(s): "
            + ", ".join(unknown)
            + ". Allowed: "
            + ", ".join(sorted(ALLOWED_STAGES))
        )
    if "all" in stages and len(stages) != 1:
        parser.error(
            "--stages all must be used alone, not mixed with other stages."
        )

    args.stages_list = stages
    return args


def main():
    args = parse_args()

    for p, label in [
        (args.train_data, "--train-data"),
        (args.val_input, "--val-input"),
        (args.runner, "--runner"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist: {p}")

    if args.test_input is not None and not args.test_input.exists():
        raise FileNotFoundError(
            f"--test-input does not exist: {args.test_input}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    current_params = dict(BASE_PARAMS)
    if args.base_params_file is not None:
        current_params.update(load_json(args.base_params_file))

    validate_online_geometry(current_params)

    save_json(
        args.output_dir / "starting_base_params.json",
        current_params,
    )

    print("=" * 88)
    print("CATCH + Optuna automated tuning")
    print("Train      :", args.train_data)
    print("Validation :", args.val_input)
    print("Test       :", args.test_input if args.test_input else "(not supplied)")
    print("Runner     :", args.runner)
    print("Output dir :", args.output_dir)
    print("Stages     :", ", ".join(args.stages_list))
    print("Trials/stage:", args.n_trials)
    print("Objective  :", args.objective)
    print("=" * 88)

    all_stage_summaries = []

    for stage in args.stages_list:
        current_params, summary = run_stage(
            args,
            stage=stage,
            current_params=current_params,
        )
        all_stage_summaries.append(summary)

        save_json(
            args.output_dir / "best_params_so_far.json",
            current_params,
        )

    save_json(
        args.output_dir / "best_params.json",
        current_params,
    )
    save_json(
        args.output_dir / "stage_summaries.json",
        all_stage_summaries,
    )

    final_result = run_final(args, current_params)

    print("\n" + "=" * 88)
    print("[DONE] Automated tuning finished.")
    print("Best params :", args.output_dir / "best_params.json")
    print("Study DB    :", args.output_dir / "optuna_studies.db")
    print("Final dir   :", args.output_dir / "final")
    print("Validation F1:", f"{final_result['validation_metrics']['f1']:.6f}")

    if "test_metrics" in final_result:
        print("Test F1      :", f"{final_result['test_metrics']['f1']:.6f}")
        print("Test PR-AUC  :", f"{final_result['test_metrics']['pr_auc']:.6f}")
        print("Test Recall  :", f"{final_result['test_metrics']['recall']:.6f}")
        print("Test FPR     :", f"{final_result['test_metrics']['fpr']:.6f}")

    print("=" * 88)


if __name__ == "__main__":
    main()
