# -*- coding: utf-8 -*-
"""
Evaluation helpers for the CATCH block-online runner.

Design goals
------------
1. Evaluation is completely separated from CATCH model training/inference.
2. Only formally scored online points are used for point-level metrics.
3. Ground-truth labels never participate in model fitting or threshold calibration.
4. When labels are absent, the runner simply skips classification evaluation.
5. A one-row CSV can be appended across experiments for hyperparameter comparison.

Metrics
-------
Point level:
    Accuracy, Precision, Recall, F1, Specificity, FPR, FNR,
    ROC-AUC, PR-AUC, TP/TN/FP/FN.

Event level:
    Number of contiguous ground-truth anomaly events,
    evaluable events, detected events, event recall,
    aligned detection delay, and actual online release delay.

Runtime:
    score-release delay, model inference time, trigger end-to-end time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:
    raise ImportError(
        "Evaluation requires scikit-learn. Install with: "
        "pip install scikit-learn"
    ) from exc


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _finite_percentile(values, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _finite_max(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.max())


def _binary_labels(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy()
    if not np.isfinite(values.astype(float)).all():
        raise ValueError("ground-truth/prediction labels contain NaN or Inf.")
    # Keep compatibility with anomaly datasets using any non-zero anomaly label.
    return (values != 0).astype(np.int64)


def _contiguous_positive_segments(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive 0-based [start, end] segments where binary == 1."""
    binary = np.asarray(binary, dtype=np.int8)
    if binary.size == 0:
        return []

    padded = np.r_[0, binary, 0]
    change = np.diff(padded)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _config_dict(model) -> dict[str, Any]:
    try:
        return dict(vars(model.config))
    except Exception:
        return {}


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def evaluate_online_output(
    full_output: pd.DataFrame,
    model,
    results,
    *,
    sample_period_seconds: Optional[float] = None,
    calibration_ratio: Optional[float] = None,
    input_path: Optional[str] = None,
    bundle_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Evaluate the exact formal online output.

    Important:
    - point metrics use ONLY status == "scored";
    - warm-up / future-context rows are excluded;
    - labels are used only here, after all model inference is complete.
    """
    metrics: dict[str, Any] = {}

    cfg = _config_dict(model)
    metrics["input_file"] = None if input_path is None else str(input_path)
    metrics["bundle_file"] = None if bundle_path is None else str(bundle_path)
    metrics["seq_len"] = int(model.config.seq_len)
    metrics["online_stride"] = int(model.online_stride)
    metrics["inference_patch_size"] = int(model.config.inference_patch_size)
    metrics["inference_patch_stride"] = int(model.config.inference_patch_stride)
    metrics["right_context"] = int(getattr(model, "right_context", -1))
    metrics["threshold"] = (
        float(model.online_threshold)
        if model.online_threshold is not None
        else float("nan")
    )
    metrics["calibration_ratio"] = (
        float(calibration_ratio)
        if calibration_ratio is not None
        else float("nan")
    )

    # Frequently tuned model parameters are flattened into the comparison row.
    for name in (
        "lr", "Mlr", "e_layers", "n_heads", "cf_dim", "d_ff", "d_model",
        "head_dim", "dropout", "head_dropout", "auxi_lambda",
        "score_lambda", "regular_lambda", "temperature", "patch_stride",
        "patch_size", "dc_lambda", "num_epochs", "batch_size", "patience",
        "anomaly_ratio", "pct_start",
    ):
        if name in cfg:
            metrics[name] = _jsonable(cfg[name])

    metrics["model_config_json"] = json.dumps(
        {k: _jsonable(v) for k, v in cfg.items()},
        ensure_ascii=False,
        sort_keys=True,
    )

    scored_mask = (
        full_output["status"].astype(str).eq("scored")
        & full_output["online_score"].notna()
    )
    metrics["total_input_samples"] = int(len(full_output))
    metrics["formally_scored_samples"] = int(scored_mask.sum())
    metrics["initial_warmup_samples"] = int(
        full_output["status"].astype(str).eq("initial_warmup").sum()
    )
    metrics["pending_future_context_samples"] = int(
        full_output["status"].astype(str).eq("pending_future_context").sum()
    )
    metrics["forward_calls"] = int(len(results))

    # Runtime metrics from trigger results.
    if results:
        model_ms = [
            r.model_inference_ms
            for r in results
            if r.model_inference_ms is not None
        ]
        e2e_ms = [
            r.trigger_end_to_end_ms
            for r in results
            if r.trigger_end_to_end_ms is not None
        ]
    else:
        model_ms, e2e_ms = [], []

    metrics["model_inference_ms_mean"] = _finite_mean(model_ms)
    metrics["model_inference_ms_p95"] = _finite_percentile(model_ms, 95)
    metrics["model_inference_ms_max"] = _finite_max(model_ms)
    metrics["trigger_e2e_ms_mean"] = _finite_mean(e2e_ms)
    metrics["trigger_e2e_ms_p95"] = _finite_percentile(e2e_ms, 95)
    metrics["trigger_e2e_ms_max"] = _finite_max(e2e_ms)

    if "detection_delay_samples" in full_output.columns:
        release_delays = pd.to_numeric(
            full_output.loc[scored_mask, "detection_delay_samples"],
            errors="coerce",
        ).to_numpy(dtype=float)
        metrics["score_release_delay_samples_mean"] = _finite_mean(release_delays)
        metrics["score_release_delay_samples_p95"] = _finite_percentile(
            release_delays, 95
        )
        metrics["score_release_delay_samples_max"] = _finite_max(release_delays)

        if sample_period_seconds is not None:
            metrics["score_release_delay_seconds_mean"] = (
                metrics["score_release_delay_samples_mean"]
                * float(sample_period_seconds)
            )
            metrics["score_release_delay_seconds_p95"] = (
                metrics["score_release_delay_samples_p95"]
                * float(sample_period_seconds)
            )
            metrics["score_release_delay_seconds_max"] = (
                metrics["score_release_delay_samples_max"]
                * float(sample_period_seconds)
            )

    if sample_period_seconds is not None:
        budget_ms = (
            int(model.online_stride)
            * float(sample_period_seconds)
            * 1000.0
        )
        metrics["trigger_budget_ms"] = budget_ms
        p95 = metrics["trigger_e2e_ms_p95"]
        metrics["realtime_p95_within_budget"] = (
            bool(np.isfinite(p95) and p95 < budget_ms)
        )

    # No labels -> runtime-only evaluation.
    if "ground_truth_label" not in full_output.columns:
        metrics["has_ground_truth"] = False
        return metrics

    eval_mask = (
        scored_mask
        & full_output["ground_truth_label"].notna()
        & full_output["pred_label"].notna()
    )
    metrics["has_ground_truth"] = True
    metrics["evaluated_scored_samples"] = int(eval_mask.sum())

    if int(eval_mask.sum()) == 0:
        return metrics

    y_true = _binary_labels(full_output.loc[eval_mask, "ground_truth_label"])
    y_pred = _binary_labels(full_output.loc[eval_mask, "pred_label"])
    y_score = pd.to_numeric(
        full_output.loc[eval_mask, "online_score"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(y_score).all():
        raise ValueError("online_score contains NaN/Inf inside evaluated rows.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    metrics.update({
        "true_normal_points": int((y_true == 0).sum()),
        "true_anomaly_points": int((y_true == 1).sum()),
        "predicted_normal_points": int((y_pred == 0).sum()),
        "predicted_anomaly_points": int((y_pred == 1).sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": _safe_div(tn, tn + fp),
        "false_positive_rate": _safe_div(fp, tn + fp),
        "false_negative_rate": _safe_div(fn, tp + fn),
    })

    if np.unique(y_true).size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    # Event-level evaluation on the original aligned timeline.
    gt_full = pd.to_numeric(
        full_output["ground_truth_label"],
        errors="coerce",
    )
    gt_valid = gt_full.notna().to_numpy()
    gt_bin = np.zeros(len(full_output), dtype=np.int8)
    gt_bin[gt_valid] = (
        gt_full.loc[gt_valid].to_numpy(dtype=float) != 0
    ).astype(np.int8)

    segments = _contiguous_positive_segments(gt_bin)
    scored_arr = scored_mask.to_numpy(dtype=bool)
    pred_arr = np.zeros(len(full_output), dtype=np.int8)
    pred_valid = full_output["pred_label"].notna().to_numpy()
    if pred_valid.any():
        pred_arr[pred_valid] = (
            pd.to_numeric(
                full_output.loc[pred_valid, "pred_label"],
                errors="raise",
            ).to_numpy(dtype=float) != 0
        ).astype(np.int8)

    evaluable_events = 0
    detected_events = 0
    aligned_delays = []
    release_delays = []

    sample_index = full_output["sample_index"].to_numpy(dtype=int)
    trigger_end = pd.to_numeric(
        full_output["trigger_end_index"],
        errors="coerce",
    ).to_numpy(dtype=float, na_value=np.nan)

    for start0, end0 in segments:
        event_rows = np.arange(start0, end0 + 1)
        scored_event_rows = event_rows[scored_arr[event_rows]]
        if scored_event_rows.size == 0:
            continue

        evaluable_events += 1
        detected_rows = scored_event_rows[pred_arr[scored_event_rows] == 1]
        if detected_rows.size == 0:
            continue

        detected_events += 1
        first_row = int(detected_rows[0])
        event_start_sample = int(sample_index[start0])
        predicted_sample = int(sample_index[first_row])

        # Aligned delay: where the first correctly flagged anomaly point lies.
        aligned_delays.append(predicted_sample - event_start_sample)

        # Release delay: when that point's score actually became available online.
        if np.isfinite(trigger_end[first_row]):
            release_delays.append(
                int(trigger_end[first_row]) - event_start_sample
            )

    metrics["ground_truth_events_total"] = int(len(segments))
    metrics["ground_truth_events_evaluable"] = int(evaluable_events)
    metrics["events_detected"] = int(detected_events)
    metrics["event_recall"] = _safe_div(detected_events, evaluable_events)

    metrics["event_aligned_delay_samples_mean"] = _finite_mean(aligned_delays)
    metrics["event_aligned_delay_samples_p95"] = _finite_percentile(
        aligned_delays, 95
    )
    metrics["event_release_delay_samples_mean"] = _finite_mean(release_delays)
    metrics["event_release_delay_samples_p95"] = _finite_percentile(
        release_delays, 95
    )

    if sample_period_seconds is not None:
        sp = float(sample_period_seconds)
        metrics["event_aligned_delay_seconds_mean"] = (
            metrics["event_aligned_delay_samples_mean"] * sp
        )
        metrics["event_aligned_delay_seconds_p95"] = (
            metrics["event_aligned_delay_samples_p95"] * sp
        )
        metrics["event_release_delay_seconds_mean"] = (
            metrics["event_release_delay_samples_mean"] * sp
        )
        metrics["event_release_delay_seconds_p95"] = (
            metrics["event_release_delay_samples_p95"] * sp
        )

    return metrics


def print_evaluation(metrics: dict[str, Any]) -> None:
    print("=" * 88)
    print("[EVALUATION]")

    if not metrics.get("has_ground_truth", False):
        print("Ground truth          : not found -> classification metrics skipped")
    else:
        print("Evaluated samples     :", metrics.get("evaluated_scored_samples", 0))
        for key, label in (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("specificity", "Specificity"),
            ("false_positive_rate", "False positive rate"),
            ("false_negative_rate", "False negative rate"),
            ("event_recall", "Event recall"),
        ):
            value = metrics.get(key, float("nan"))
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                print(f"{label:<22}: {float(value):.6f}")
            else:
                print(f"{label:<22}: NA")

        print(
            "Confusion matrix      : "
            f"TN={metrics.get('tn', 0)}, FP={metrics.get('fp', 0)}, "
            f"FN={metrics.get('fn', 0)}, TP={metrics.get('tp', 0)}"
        )
        print(
            "Events                : "
            f"{metrics.get('events_detected', 0)}/"
            f"{metrics.get('ground_truth_events_evaluable', 0)} detected "
            f"(total GT={metrics.get('ground_truth_events_total', 0)})"
        )

        d = metrics.get("event_release_delay_samples_mean", float("nan"))
        if isinstance(d, (int, float, np.number)) and np.isfinite(d):
            print(f"Mean event release delay: {float(d):.4f} samples")

    p95 = metrics.get("trigger_e2e_ms_p95", float("nan"))
    if isinstance(p95, (int, float, np.number)) and np.isfinite(p95):
        print(f"P95 trigger E2E       : {float(p95):.4f} ms")

    print("=" * 88)


def save_evaluation(
    metrics: dict[str, Any],
    path,
    *,
    append: bool = False,
) -> Path:
    """
    Save one experiment row.

    append=True is intended for hyperparameter comparison:
    all runs are appended into the same CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([metrics])

    if append and path.exists():
        old = pd.read_csv(path)

        # Union columns without losing older experiment fields.
        all_cols = list(dict.fromkeys(list(old.columns) + list(row.columns)))
        old = old.reindex(columns=all_cols)
        row = row.reindex(columns=all_cols)
        out = pd.concat([old, row], ignore_index=True)
    else:
        out = row

    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path
