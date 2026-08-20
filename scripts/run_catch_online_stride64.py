# -*- coding: utf-8 -*-
"""
CATCH block-online runner with:
1) original CATCH training/inference behavior preserved;
2) full evaluation module;
3) adjustable online geometry.

Backward-compatible defaults:
    seq_len / window           = 192
    online_stride              = 64
    inference_patch_size       = 32
    inference_patch_stride     = 1

Recommended: put ALL model + online geometry parameters in
--model-hyper-params JSON, including:
    seq_len
    online_stride
    inference_patch_size
    inference_patch_stride

Legacy --window / --stride remain available as direct overrides.

Example:
python ./scripts/run_catch_online_stride64.py \
  --train-data normal.csv \
  --input test.csv \
  --model-hyper-params \
  '{"seq_len":192,"online_stride":64,"inference_patch_size":32,
    "inference_patch_stride":1,"patch_size":16,"patch_stride":8,
    "lr":0.0005,"Mlr":1e-05,"d_model":256,"batch_size":32,
    "num_epochs":5,"score_lambda":0.05,"dc_lambda":0.1,
    "auxi_lambda":0.05,"anomaly_ratio":1.0}' \
  --bundle ./checkpoints/exp01.pt \
  --output ./results/exp01_points.csv \
  --evaluation-output ./results/hparam_comparison.csv \
  --evaluation-append
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.baselines.catch.CATCH_online import CATCHOnline
from scripts.catch_online_evaluation import (
    evaluate_online_output,
    print_evaluation,
    save_evaluation,
)


LABEL_CANDIDATES = {
    "label",
    "labels",
    "is_anomaly",
    "anomaly",
}

TIME_CANDIDATES = {
    "date",
    "time",
    "timestamp",
    "index",
}


# ----------------------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------------------
SUPPORTED_MODEL_HYPER_PARAMS = {
    "lr",
    "Mlr",
    "e_layers",
    "n_heads",
    "cf_dim",
    "d_ff",
    "d_model",
    "head_dim",
    "individual",
    "dropout",
    "head_dropout",
    "auxi_loss",
    "auxi_type",
    "auxi_mode",
    "auxi_lambda",
    "score_lambda",
    "regular_lambda",
    "temperature",
    "patch_stride",
    "patch_size",
    "inference_patch_stride",
    "inference_patch_size",
    "dc_lambda",
    "module_first",
    "mask",
    "pretrained_model",
    "num_epochs",
    "batch_size",
    "patience",
    "anomaly_ratio",
    "seq_len",
    "pct_start",
    "revin",
    "affine",
    "subtract_last",
    "lradj",

    # Runner-level online geometry; passed separately to CATCHOnline.
    "online_stride",
}


def parse_model_hyper_params(text: Optional[str]) -> dict:
    if text is None or not str(text).strip():
        return {}

    try:
        params = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--model-hyper-params must be valid JSON, for example: "
            "'{\"lr\":0.0005,\"d_model\":256,\"seq_len\":192,"
            "\"online_stride\":64}'."
        ) from exc

    if not isinstance(params, dict):
        raise ValueError(
            "--model-hyper-params must decode to one JSON object/dict."
        )

    unknown = sorted(
        str(k)
        for k in params
        if k not in SUPPORTED_MODEL_HYPER_PARAMS
    )
    if unknown:
        raise ValueError(
            "Unknown CATCH/online hyperparameter(s): "
            + ", ".join(unknown)
        )

    return dict(params)


def _to_int(name: str, value, minimum: Optional[int] = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def resolve_training_hyper_params(args) -> None:
    """
    Precedence:
        direct CLI override > JSON > backward-compatible default
    """
    params = parse_model_hyper_params(args.model_hyper_params)

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else params.get("batch_size", 32)
    )
    num_epochs = (
        args.num_epochs
        if args.num_epochs is not None
        else params.get("num_epochs", 3)
    )
    anomaly_ratio = (
        args.anomaly_ratio
        if args.anomaly_ratio is not None
        else params.get("anomaly_ratio", 1.0)
    )

    seq_len = (
        args.window
        if args.window is not None
        else params.get("seq_len", 192)
    )
    online_stride = (
        args.stride
        if args.stride is not None
        else params.get("online_stride", 64)
    )
    inference_patch_size = params.get(
        "inference_patch_size",
        32,
    )
    inference_patch_stride = params.get(
        "inference_patch_stride",
        1,
    )

    batch_size = _to_int("batch_size", batch_size, 8)
    num_epochs = _to_int("num_epochs", num_epochs, 1)
    seq_len = _to_int("seq_len", seq_len, 1)
    online_stride = _to_int("online_stride", online_stride, 1)
    inference_patch_size = _to_int(
        "inference_patch_size",
        inference_patch_size,
        1,
    )
    inference_patch_stride = _to_int(
        "inference_patch_stride",
        inference_patch_stride,
        1,
    )

    if isinstance(anomaly_ratio, (list, tuple, dict)):
        raise ValueError(
            "Online threshold calibration requires one scalar "
            "anomaly_ratio, not a list/dict."
        )
    try:
        anomaly_ratio = float(anomaly_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "anomaly_ratio must be a numeric percentage."
        ) from exc

    if not (0.0 < anomaly_ratio < 100.0):
        raise ValueError(
            "anomaly_ratio must be in (0, 100)."
        )

    if online_stride > seq_len:
        raise ValueError(
            "online_stride must be <= seq_len."
        )
    if inference_patch_size > seq_len:
        raise ValueError(
            "inference_patch_size must be <= seq_len."
        )
    if inference_patch_stride > inference_patch_size:
        raise ValueError(
            "inference_patch_stride must be <= inference_patch_size."
        )

    minimum_seq_len = (
        online_stride
        + 2 * (inference_patch_size - 1)
    )
    if seq_len < minimum_seq_len:
        raise ValueError(
            "Boundary-safe online geometry is invalid: "
            f"seq_len={seq_len}, online_stride={online_stride}, "
            f"inference_patch_size={inference_patch_size}. "
            f"Require seq_len >= {minimum_seq_len}."
        )

    args.model_hyper_params_dict = params
    args.batch_size = batch_size
    args.num_epochs = num_epochs
    args.anomaly_ratio = anomaly_ratio
    args.seq_len = seq_len
    args.online_stride = online_stride
    args.inference_patch_size = inference_patch_size
    args.inference_patch_stride = inference_patch_stride


def build_effective_model_hyper_params(
    args,
    batch_size: int,
) -> dict:
    params = dict(args.model_hyper_params_dict)

    # online_stride belongs to CATCHOnline wrapper, not TransformerConfig.
    params.pop("online_stride", None)

    params["batch_size"] = int(batch_size)
    params["num_epochs"] = int(args.num_epochs)
    params["anomaly_ratio"] = float(args.anomaly_ratio)

    # Online geometry is no longer force-fixed; resolved values are used.
    params["seq_len"] = int(args.seq_len)
    params["inference_patch_size"] = int(
        args.inference_patch_size
    )
    params["inference_patch_stride"] = int(
        args.inference_patch_stride
    )

    return params


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# File loading
# ----------------------------------------------------------------------
def _read_table(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported file type: {suffix}. "
        "Use CSV, XLSX, or XLS."
    )


def _find_case_insensitive(
    columns,
    candidates,
):
    mapping = {
        str(c).strip().lower(): c
        for c in columns
    }

    for name in candidates:
        key = str(name).strip().lower()
        if key in mapping:
            return mapping[key]

    return None


def validate_strict_time_order(
    timestamps: pd.Series,
    source_name: str,
) -> None:
    ts = pd.Series(
        timestamps
    ).reset_index(drop=True)

    if ts.isna().any():
        raise ValueError(
            f"{source_name}: timestamp/date contains NaN."
        )

    if len(ts) <= 1:
        return

    numeric = pd.to_numeric(
        ts,
        errors="coerce",
    )

    if numeric.notna().all():
        values = numeric.to_numpy(
            dtype=np.float64
        )
        diffs = np.diff(values)

        if not np.all(diffs > 0):
            bad = np.where(diffs <= 0)[0][:5]
            raise ValueError(
                f"{source_name}: date/time must be strictly increasing. "
                f"Problem near row indices {(bad + 1).tolist()}."
            )
        return

    dt = pd.to_datetime(
        ts,
        errors="coerce",
    )

    if dt.notna().all():
        values = dt.astype(
            "int64"
        ).to_numpy()
        diffs = np.diff(values)

        if not np.all(diffs > 0):
            raise ValueError(
                f"{source_name}: datetime must be strictly increasing."
            )
        return

    idx = pd.Index(
        ts.astype(str)
    )

    if (
        not idx.is_monotonic_increasing
        or idx.has_duplicates
    ):
        raise ValueError(
            f"{source_name}: date/time is not strictly increasing."
        )


def load_timeseries_file(
    path,
    *,
    time_col: Optional[str] = None,
    label_col: Optional[str] = None,
):
    raw = _read_table(path)

    if raw.empty:
        raise ValueError(
            f"Input file is empty: {path}"
        )

    lower_cols = {
        str(c).strip().lower(): c
        for c in raw.columns
    }

    # ----------------------------------------------------------
    # A) CATCH long format: date,data,cols
    # ----------------------------------------------------------
    if {
        "date",
        "data",
        "cols",
    }.issubset(lower_cols):

        date_c = lower_cols["date"]
        data_c = lower_cols["data"]
        cols_c = lower_cols["cols"]

        tmp = raw[
            [date_c, data_c, cols_c]
        ].copy()

        tmp[cols_c] = (
            tmp[cols_c]
            .astype(str)
            .str.strip()
        )

        dup = tmp.duplicated(
            subset=[date_c, cols_c],
            keep=False,
        )

        if dup.any():
            examples = tmp.loc[
                dup,
                [date_c, cols_c],
            ].head(10)

            raise ValueError(
                "Long-format input contains duplicate "
                "(date, cols) pairs. Examples:\n"
                + examples.to_string(index=False)
            )

        date_order = pd.Index(
            tmp[date_c]
            .drop_duplicates()
            .tolist()
        )

        timestamps = pd.Series(
            date_order.to_list(),
            name="date",
        )

        validate_strict_time_order(
            timestamps,
            source_name=str(path),
        )

        variable_order = (
            tmp[cols_c]
            .drop_duplicates()
            .tolist()
        )

        wide = tmp.pivot(
            index=date_c,
            columns=cols_c,
            values=data_c,
        )

        wide = wide.reindex(
            date_order
        )

        wide = wide.reindex(
            columns=variable_order
        )

        detected_label = None

        if label_col is not None:
            for c in wide.columns:
                if (
                    str(c).strip().lower()
                    == label_col.strip().lower()
                ):
                    detected_label = c
                    break
        else:
            for c in wide.columns:
                if (
                    str(c).strip().lower()
                    in LABEL_CANDIDATES
                ):
                    detected_label = c
                    break

        ground_truth = None

        if detected_label is not None:
            ground_truth = pd.to_numeric(
                wide[detected_label],
                errors="raise",
            ).reset_index(drop=True)

            wide = wide.drop(
                columns=[detected_label]
            )

        if wide.shape[1] == 0:
            raise ValueError(
                "No feature columns remain after removing label."
            )

        for c in wide.columns:
            wide[c] = pd.to_numeric(
                wide[c],
                errors="raise",
            )

        if wide.isna().any().any():
            bad = wide.columns[
                wide.isna().any()
            ].tolist()

            raise ValueError(
                "Missing values exist after long->wide conversion "
                "in feature columns: "
                + ", ".join(map(str, bad))
            )

        features = wide.reset_index(
            drop=True
        )

        features.columns = [
            str(c)
            for c in features.columns
        ]

        return (
            features,
            timestamps,
            ground_truth,
            "catch_long",
        )

    # ----------------------------------------------------------
    # B) ordinary wide format
    # ----------------------------------------------------------
    wide = raw.copy()

    if time_col is not None:
        if time_col in wide.columns:
            time_c = time_col
        else:
            time_c = _find_case_insensitive(
                wide.columns,
                {time_col},
            )
            if time_c is None:
                raise ValueError(
                    f"--time-col '{time_col}' not found."
                )
    else:
        time_c = _find_case_insensitive(
            wide.columns,
            TIME_CANDIDATES,
        )

    if time_c is None:
        timestamps = pd.Series(
            np.arange(
                1,
                len(wide) + 1,
            ),
            name="date",
        )
    else:
        timestamps = wide[
            time_c
        ].reset_index(drop=True)

        validate_strict_time_order(
            timestamps,
            source_name=str(path),
        )

        wide = wide.drop(
            columns=[time_c]
        )

    if label_col is not None:
        if label_col in wide.columns:
            label_c = label_col
        else:
            label_c = _find_case_insensitive(
                wide.columns,
                {label_col},
            )
    else:
        label_c = _find_case_insensitive(
            wide.columns,
            LABEL_CANDIDATES,
        )

    ground_truth = None

    if label_c is not None:
        ground_truth = pd.to_numeric(
            wide[label_c],
            errors="raise",
        ).reset_index(drop=True)

        wide = wide.drop(
            columns=[label_c]
        )

    if wide.shape[1] == 0:
        raise ValueError(
            "No feature columns found."
        )

    for c in wide.columns:
        wide[c] = pd.to_numeric(
            wide[c],
            errors="raise",
        )

    if wide.isna().any().any():
        bad = wide.columns[
            wide.isna().any()
        ].tolist()

        raise ValueError(
            "NaN exists in feature columns: "
            + ", ".join(map(str, bad))
        )

    features = wide.reset_index(
        drop=True
    )

    features.columns = [
        str(c)
        for c in features.columns
    ]

    return (
        features,
        timestamps,
        ground_truth,
        "wide",
    )


def reorder_features_for_bundle(
    features,
    expected_features,
):
    if not expected_features:
        return features

    expected = [
        str(c)
        for c in expected_features
    ]
    actual = [
        str(c)
        for c in features.columns
    ]

    missing = [
        c
        for c in expected
        if c not in actual
    ]

    if missing:
        raise ValueError(
            "Monitoring input is missing model features: "
            + ", ".join(missing)
        )

    extra = [
        c
        for c in actual
        if c not in expected
    ]

    if extra:
        print(
            "[WARNING] Extra input columns are ignored: "
            + ", ".join(extra)
        )

    return features.loc[
        :,
        expected,
    ].copy()


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def build_full_output(
    model,
    timestamps,
    ground_truth,
    results,
    sample_period_seconds: Optional[float],
):
    n = len(timestamps)

    out = pd.DataFrame({
        "sample_index": np.arange(
            1,
            n + 1,
            dtype=np.int64,
        ),
        "date": timestamps.values,
        "online_score": np.full(
            n,
            np.nan,
            dtype=np.float64,
        ),
        "pred_label": pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        "status": np.array(
            ["pending_future_context"] * n,
            dtype=object,
        ),
        "trigger_end_index": pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        "detection_delay_samples": pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        "detection_delay_seconds": np.full(
            n,
            np.nan,
            dtype=np.float64,
        ),
        "batch_alarm": pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        "model_inference_ms": np.full(
            n,
            np.nan,
            dtype=np.float64,
        ),
        "trigger_end_to_end_ms": np.full(
            n,
            np.nan,
            dtype=np.float64,
        ),
    })

    if model.initial_warmup_count > 0:
        out.loc[
            0:
            model.initial_warmup_count - 1,
            "status",
        ] = "initial_warmup"

    if ground_truth is not None:
        out["ground_truth_label"] = (
            pd.to_numeric(
                ground_truth,
                errors="raise",
            )
            .astype("Int64")
        )

    for r in results:
        start0 = r.score_start_index - 1
        end0 = r.score_end_index

        expected_len = end0 - start0

        if expected_len != len(r.scores):
            raise RuntimeError(
                "Score/index alignment error: "
                f"expected {expected_len}, got {len(r.scores)}."
            )

        rows = np.arange(
            start0,
            end0,
        )

        out.loc[
            rows,
            "online_score",
        ] = r.scores

        out.loc[
            rows,
            "status",
        ] = "scored"

        out.loc[
            rows,
            "trigger_end_index",
        ] = r.window_end_index

        delays = (
            r.window_end_index
            - out.loc[
                rows,
                "sample_index",
            ].to_numpy()
        )

        out.loc[
            rows,
            "detection_delay_samples",
        ] = delays

        if sample_period_seconds is not None:
            out.loc[
                rows,
                "detection_delay_seconds",
            ] = (
                delays
                * float(sample_period_seconds)
            )

        out.loc[
            rows,
            "model_inference_ms",
        ] = r.model_inference_ms

        out.loc[
            rows,
            "trigger_end_to_end_ms",
        ] = r.trigger_end_to_end_ms

        if r.labels is not None:
            out.loc[
                rows,
                "pred_label",
            ] = r.labels.astype(
                np.int64
            )

            out.loc[
                rows,
                "batch_alarm",
            ] = int(bool(r.alarm))

    return out


# ----------------------------------------------------------------------
# Training with batch-size OOM fallback
# ----------------------------------------------------------------------
def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(
        exc,
        torch.OutOfMemoryError,
    ):
        return True

    text = str(exc).lower()

    return (
        "cuda out of memory" in text
        or (
            "out of memory" in text
            and "cuda" in text
        )
    )


def paper_batch_candidates(
    requested_batch_size: int,
):
    b = int(requested_batch_size)

    if b < 8:
        raise ValueError(
            "Training batch size must be >= 8."
        )

    candidates = []

    while b >= 8:
        if b not in candidates:
            candidates.append(b)

        if b == 8:
            break

        b = max(
            8,
            b // 2,
        )

    return candidates


def train_with_oom_fallback(
    train_features,
    args,
):
    last_oom = None

    for batch_size in paper_batch_candidates(
        args.batch_size
    ):
        print(
            f"[TRAIN] trying batch_size={batch_size}"
        )

        set_global_seed(
            args.seed
        )

        model_hyper_params = build_effective_model_hyper_params(
            args,
            batch_size=batch_size,
        )

        model = CATCHOnline(
            online_stride=args.online_stride,
            inference_mask_seed=(
                None
                if args.stochastic_mask
                else args.mask_seed
            ),
            **model_hyper_params,
        )

        model.device = torch.device(
            resolve_device(
                args.device
            )
        )

        try:
            threshold = model.fit_for_online(
                train_features,
                calibration_ratio=args.calibration_ratio,
                anomaly_ratio_percent=args.anomaly_ratio,
            )

            return (
                model,
                threshold,
                batch_size,
            )

        except BaseException as exc:
            if not is_cuda_oom(exc):
                raise

            last_oom = exc

            print(
                f"[OOM] batch_size={batch_size} failed."
            )

            del model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if batch_size == 8:
                break

            print(
                "[OOM] retrying with half batch size..."
            )

    raise RuntimeError(
        "CUDA OOM even at batch_size=8. "
        "Reduce model/data load or use a larger GPU."
    ) from last_oom


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def resolve_device(name):
    if name == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "--device cuda was requested, "
            "but CUDA is unavailable."
        )

    return name


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CATCH block-online monitor with adjustable geometry "
            "and evaluation."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Monitoring/test CSV/XLSX. "
            "Supports CATCH long or ordinary wide format."
        ),
    )

    parser.add_argument(
        "--train-data",
        default=None,
        help=(
            "Historical NORMAL data for first-time training. "
            "Omit after bundle has been created."
        ),
    )

    # Keep previous default path unchanged for compatibility.
    parser.add_argument(
        "--bundle",
        default=(
            "./checkpoints/"
            "catch_online_stride64_final.pt"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--time-col",
        default=None,
    )

    parser.add_argument(
        "--label-col",
        default=None,
    )

    # Legacy direct overrides. None is essential so JSON can control them.
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help=(
            "Optional direct seq_len override. "
            "If omitted: JSON seq_len, else default=192."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Optional direct online_stride override. "
            "If omitted: JSON online_stride, else default=64."
        ),
    )

    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=None,
        help=(
            "Threshold anomaly percentage. "
            "Direct option > JSON anomaly_ratio > default 1.0."
        ),
    )

    parser.add_argument(
        "--model-hyper-params",
        default=None,
        help=(
            "CATCH + online hyperparameters as one JSON object. "
            "seq_len, online_stride, inference_patch_size and "
            "inference_patch_stride are now adjustable here."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Direct batch_size override. "
            "Direct > JSON > default 32. "
            "CUDA OOM halves down to 8."
        ),
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help=(
            "Direct num_epochs override. "
            "Direct > JSON > default 3."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--mask-seed",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--stochastic-mask",
        action="store_true",
    )

    parser.add_argument(
        "--sample-period",
        type=float,
        default=None,
        help=(
            "Sampling period in seconds; used for latency and "
            "real-time budget reporting."
        ),
    )

    parser.add_argument(
        "--repeatability-check",
        type=int,
        default=0,
    )

    # New evaluation options.
    parser.add_argument(
        "--evaluation-output",
        default=None,
        help=(
            "One-row evaluation CSV. Default: beside point output, "
            "suffix _evaluation.csv."
        ),
    )

    parser.add_argument(
        "--evaluation-append",
        action="store_true",
        help=(
            "Append this run to evaluation CSV. "
            "Use one shared file to compare hyperparameter experiments."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    resolve_training_hyper_params(args)

    if (
        args.sample_period is not None
        and args.sample_period <= 0
    ):
        raise ValueError(
            "--sample-period must be > 0."
        )

    input_path = Path(
        args.input
    )

    bundle_path = Path(
        args.bundle
    )

    # Keep previous default output naming unchanged.
    if args.output is None:
        output_path = input_path.with_name(
            input_path.stem
            + "_CATCH_online_stride64_FINAL.csv"
        )
    else:
        output_path = Path(
            args.output
        )

    if args.evaluation_output is None:
        evaluation_path = output_path.with_name(
            output_path.stem
            + "_evaluation.csv"
        )
    else:
        evaluation_path = Path(
            args.evaluation_output
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    evaluation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("CATCH BLOCK-ONLINE MONITOR + EVALUATION")
    print(
        "Requested geometry: "
        f"seq_len={args.seq_len} | "
        f"online_stride={args.online_stride} | "
        f"inference_patch_size={args.inference_patch_size} | "
        f"inference_patch_stride={args.inference_patch_stride}"
    )
    print("Input :", input_path)
    print("Bundle:", bundle_path)
    print("Output:", output_path)
    print("Eval  :", evaluation_path)
    print("=" * 88)

    # ----------------------------------------------------------
    # First-time training
    # ----------------------------------------------------------
    if args.train_data is not None:
        (
            train_features,
            _train_ts,
            train_gt,
            train_format,
        ) = load_timeseries_file(
            args.train_data,
            time_col=args.time_col,
            label_col=args.label_col,
        )

        if train_gt is not None:
            nonzero = int(
                (
                    pd.to_numeric(
                        train_gt
                    ) != 0
                ).sum()
            )

            if nonzero > 0:
                raise ValueError(
                    f"--train-data contains {nonzero} "
                    "non-zero labels. "
                    "Training data must be normal."
                )

        print(
            f"[TRAIN] format={train_format}"
        )
        print(
            f"[TRAIN] rows={len(train_features)}"
        )
        print(
            f"[TRAIN] features={train_features.shape[1]}"
        )
        print(
            "[TRAIN] feature order:",
            list(train_features.columns),
        )

        effective_preview = build_effective_model_hyper_params(
            args,
            batch_size=args.batch_size,
        )
        effective_preview["online_stride"] = args.online_stride
        print(
            "[TRAIN] effective CATCH/online hyperparameters:",
            json.dumps(
                effective_preview,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        (
            model,
            threshold,
            actual_batch_size,
        ) = train_with_oom_fallback(
            train_features,
            args,
        )

        model.feature_names = list(
            train_features.columns
        )

        model.save_online_bundle(
            bundle_path
        )

        print(
            f"[TRAIN] successful batch_size="
            f"{actual_batch_size}"
        )
        print(
            f"[TRAIN] fixed online threshold="
            f"{threshold:.12g}"
        )
        print(
            "[TRAIN] final bundle saved:",
            bundle_path,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # ----------------------------------------------------------
    # Load bundle
    # ----------------------------------------------------------
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle not found: {bundle_path}\n"
            "Run once with --train-data NORMAL_DATA.csv."
        )

    model = CATCHOnline.load_online_bundle(
        bundle_path,
        device=resolve_device(
            args.device
        ),
    )

    # No fixed-value rejection here: geometry comes from the bundle.
    print("[MODEL] device =", model.device)
    print("[MODEL] seq_len =", model.config.seq_len)
    print(
        "[MODEL] inference_patch_size =",
        model.config.inference_patch_size,
    )
    print(
        "[MODEL] inference_patch_stride =",
        model.config.inference_patch_stride,
    )
    print(
        "[MODEL] online_stride =",
        model.online_stride,
    )
    print(
        "[MODEL] right_context =",
        model.right_context,
    )
    print(
        "[MODEL] first formal score index =",
        model.first_formal_score_index,
    )
    print(
        "[MODEL] formal delay range (samples) =",
        f"{model.formal_score_delay_min_samples}"
        f"..{model.formal_score_delay_max_samples}",
    )
    print(
        "[MODEL] threshold =",
        model.online_threshold,
    )
    print(
        "[MODEL] expected feature order =",
        model.feature_names,
    )

    # ----------------------------------------------------------
    # Monitoring/test input
    # ----------------------------------------------------------
    (
        features,
        timestamps,
        ground_truth,
        input_format,
    ) = load_timeseries_file(
        input_path,
        time_col=args.time_col,
        label_col=args.label_col,
    )

    features = reorder_features_for_bundle(
        features,
        model.feature_names,
    )

    print(
        f"[INPUT] format={input_format}"
    )
    print(
        f"[INPUT] rows={len(features)}"
    )
    print(
        f"[INPUT] features={features.shape[1]}"
    )

    seq_len = int(model.config.seq_len)
    if len(features) < seq_len:
        raise ValueError(
            f"Monitoring input has only {len(features)} rows; "
            f"at least seq_len={seq_len} are required."
        )

    # Optional repeatability check now follows actual bundle seq_len.
    if args.repeatability_check >= 2:
        first_raw_window = (
            features.iloc[:seq_len]
            .to_numpy(dtype=np.float32)
        )

        diag = model.repeatability_check(
            first_raw_window,
            repeats=args.repeatability_check,
        )

        print(
            "[REPEATABILITY]",
            diag,
        )

    # ----------------------------------------------------------
    # Exact online replay
    # ----------------------------------------------------------
    results = model.replay_online(
        features,
        timestamps=timestamps.tolist(),
        reset_before=True,
    )

    full_output = build_full_output(
        model,
        timestamps,
        ground_truth,
        results,
        sample_period_seconds=args.sample_period,
    )

    full_output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------------------
    # Existing runtime summary
    # ----------------------------------------------------------
    scored = (
        full_output["status"]
        == "scored"
    )

    warmup = (
        full_output["status"]
        == "initial_warmup"
    )

    pending = (
        full_output["status"]
        == "pending_future_context"
    )

    print("=" * 88)
    print("[DONE]")
    print(
        "Total input samples :",
        len(full_output),
    )
    print(
        "Initial warm-up     :",
        int(warmup.sum()),
    )
    print(
        "Formally scored     :",
        int(scored.sum()),
    )
    print(
        "Pending/context     :",
        int(pending.sum()),
    )
    print(
        "CATCH forward calls :",
        len(results),
    )

    if results:
        model_ms = np.asarray(
            [
                r.model_inference_ms
                for r in results
            ],
            dtype=float,
        )

        e2e_ms = np.asarray(
            [
                r.trigger_end_to_end_ms
                for r in results
            ],
            dtype=float,
        )

        print(
            "Mean model ms       :",
            f"{model_ms.mean():.4f}",
        )
        print(
            "P95 model ms        :",
            f"{np.percentile(model_ms, 95):.4f}",
        )
        print(
            "Max model ms        :",
            f"{model_ms.max():.4f}",
        )

        print(
            "Mean trigger E2E ms :",
            f"{e2e_ms.mean():.4f}",
        )
        print(
            "P95 trigger E2E ms  :",
            f"{np.percentile(e2e_ms, 95):.4f}",
        )
        print(
            "Max trigger E2E ms  :",
            f"{e2e_ms.max():.4f}",
        )

        if args.sample_period is not None:
            trigger_budget_ms = (
                model.online_stride
                * args.sample_period
                * 1000.0
            )

            print(
                "Trigger time budget :",
                f"{trigger_budget_ms:.4f} ms",
            )

            if (
                np.percentile(e2e_ms, 95)
                < trigger_budget_ms
            ):
                print(
                    "[REALTIME] P95 trigger compute time "
                    "is below the stride interval."
                )
            else:
                print(
                    "[REALTIME WARNING] P95 trigger compute time "
                    "is NOT below the stride interval."
                )

    if (
        model.online_threshold is not None
        and int(scored.sum()) > 0
    ):
        pred = full_output.loc[
            scored,
            "pred_label",
        ].astype("Int64")

        print(
            "Predicted anomaly points:",
            int(
                (pred == 1).sum()
            ),
        )

    print(
        "Output CSV:",
        output_path,
    )
    print("=" * 88)

    # ----------------------------------------------------------
    # NEW: evaluation
    # ----------------------------------------------------------
    metrics = evaluate_online_output(
        full_output,
        model,
        results,
        sample_period_seconds=args.sample_period,
        calibration_ratio=args.calibration_ratio,
        input_path=str(input_path),
        bundle_path=str(bundle_path),
    )

    print_evaluation(metrics)

    saved_eval = save_evaluation(
        metrics,
        evaluation_path,
        append=args.evaluation_append,
    )

    print(
        "[EVALUATION] CSV saved:",
        saved_eval,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "\n[ERROR]",
            str(exc),
            file=sys.stderr,
        )
        raise
