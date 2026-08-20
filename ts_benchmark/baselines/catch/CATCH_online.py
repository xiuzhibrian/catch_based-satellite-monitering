# -*- coding: utf-8 -*-
"""
CATCH block-online adaptation with adjustable online geometry.

The original CATCH network, forward(), training flow, frequency_criterion,
scaler, EarlyStopping, and score formula remain unchanged.

Default geometry (backward compatible):
    seq_len = 192
    online_stride = 64
    inference_patch_size = 32
    inference_patch_stride = 1

All four can now be changed, subject to validity checks.

For boundary-safe formal score release this wrapper conservatively requires:
    seq_len >= online_stride + 2 * (inference_patch_size - 1)

and:
    1 <= inference_patch_stride <= inference_patch_size

The release block is still the latest contiguous online_stride-sized block
that has a full inference_patch_size-1 context on both sides.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.catch.CATCH import CATCH
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion


@dataclass
class OnlineScoreBatch:
    """One formal block of online scores."""

    window_start_index: int
    window_end_index: int
    score_start_index: int
    score_end_index: int
    scores: np.ndarray
    latest_score: float
    timestamps: Optional[list[Any]] = None
    labels: Optional[np.ndarray] = None
    alarm: Optional[bool] = None
    model_inference_ms: Optional[float] = None
    trigger_end_to_end_ms: Optional[float] = None


class CATCHOnline(CATCH):
    """
    Paper-aware block-online wrapper for CATCH.

    The wrapper changes only:
      1) when a full CATCH window is evaluated;
      2) which point scores are formally released online;
      3) how an online-safe threshold is calibrated.

    Model internals remain the author's implementation.
    """

    BUNDLE_VERSION = 2

    def __init__(
        self,
        online_stride: int = 64,
        online_threshold: Optional[float] = None,
        inference_mask_seed: Optional[int] = 2025,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.online_stride = int(online_stride)
        self.online_threshold = (
            None if online_threshold is None else float(online_threshold)
        )
        self.inference_mask_seed = (
            None if inference_mask_seed is None else int(inference_mask_seed)
        )

        self.feature_names: Optional[list[str]] = None
        self._online_freq_criterion = None
        self._online_runtime_ready = False

        self._validate_online_design()
        self._reset_online_state()

    # ------------------------------------------------------------------
    # Online geometry
    # ------------------------------------------------------------------
    def _validate_online_design(self) -> None:
        seq_len = int(self.config.seq_len)
        stride = int(self.online_stride)
        patch_size = int(self.config.inference_patch_size)
        patch_stride = int(self.config.inference_patch_stride)

        if seq_len <= 0:
            raise ValueError("config.seq_len must be > 0.")
        if stride <= 0:
            raise ValueError("online_stride must be > 0.")
        if stride > seq_len:
            raise ValueError("online_stride must be <= seq_len.")

        if patch_size <= 0:
            raise ValueError("inference_patch_size must be > 0.")
        if patch_size > seq_len:
            raise ValueError(
                "inference_patch_size must be <= seq_len."
            )

        if patch_stride <= 0:
            raise ValueError("inference_patch_stride must be > 0.")
        if patch_stride > patch_size:
            raise ValueError(
                "inference_patch_stride must be <= inference_patch_size. "
                "Larger strides can leave points uncovered by every "
                "frequency patch and produce invalid point scores."
            )

        # Conservative boundary-safe release rule.
        # A point with p-1 raw samples on both sides cannot belong only to a
        # truncated left/right boundary region, regardless of patch stride.
        right_context = patch_size - 1

        emit_end_offset = seq_len - right_context - 1
        emit_start_offset = emit_end_offset - stride + 1

        if emit_start_offset < right_context:
            minimum = stride + 2 * (patch_size - 1)
            raise ValueError(
                "Invalid online geometry for boundary-safe scoring. "
                f"seq_len={seq_len}, online_stride={stride}, "
                f"inference_patch_size={patch_size}. "
                f"Require seq_len >= {minimum} for this combination."
            )

        self.right_context = int(right_context)
        self.emit_start_offset = int(emit_start_offset)
        self.emit_end_offset = int(emit_end_offset)
        self.first_formal_score_index = self.emit_start_offset + 1
        self.initial_warmup_count = self.first_formal_score_index - 1

    @property
    def formal_score_delay_min_samples(self) -> int:
        return int(self.right_context)

    @property
    def formal_score_delay_max_samples(self) -> int:
        return int(self.right_context + self.online_stride - 1)

    # ------------------------------------------------------------------
    # Streaming state
    # ------------------------------------------------------------------
    def _reset_online_state(self) -> None:
        self._online_buffer = deque(maxlen=int(self.config.seq_len))
        self._online_timestamp_buffer = deque(maxlen=int(self.config.seq_len))
        self._online_samples_seen = 0
        self._online_next_infer_at = int(self.config.seq_len)

    def reset_online(self) -> None:
        self._reset_online_state()

    def set_online_threshold(self, threshold: Optional[float]) -> None:
        self.online_threshold = (
            None if threshold is None else float(threshold)
        )

    # ------------------------------------------------------------------
    # Runtime preparation
    # ------------------------------------------------------------------
    def prepare_online_runtime(self) -> None:
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "CATCH model is not initialized. Train it or load a bundle first."
            )
        if not hasattr(self.config, "c_in"):
            raise ValueError("config.c_in is missing.")

        self.model.to(self.device)
        self.model.eval()

        if self._online_freq_criterion is None:
            self._online_freq_criterion = frequency_criterion(self.config)
            self._online_freq_criterion.to(self.device)
            self._online_freq_criterion.eval()

        self._online_runtime_ready = True

    def _ensure_runtime_ready(self) -> None:
        if not self._online_runtime_ready:
            self.prepare_online_runtime()

    def use_best_trained_weights(self) -> None:
        if not hasattr(self, "early_stopping"):
            raise ValueError(
                "No EarlyStopping state. Call detect_fit() first."
            )
        if not hasattr(self.early_stopping, "check_point"):
            raise ValueError("EarlyStopping best checkpoint is missing.")

        self.model.load_state_dict(self.early_stopping.check_point)
        self._online_runtime_ready = False
        self.prepare_online_runtime()

    # ------------------------------------------------------------------
    # Controlled inference randomness
    # ------------------------------------------------------------------
    @contextmanager
    def _controlled_inference_rng(self):
        if self.inference_mask_seed is None:
            yield
            return

        if self.device.type == "cuda":
            device_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            devices = [device_index]
        else:
            devices = []

        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(self.inference_mask_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(self.inference_mask_seed)
            yield

    # ------------------------------------------------------------------
    # Core score calculation
    # ------------------------------------------------------------------
    def _validate_sample(self, sample: Sequence[float]) -> np.ndarray:
        x = np.asarray(sample, dtype=np.float32)

        if x.ndim != 1:
            raise ValueError(
                f"Each online sample must be 1-D [C], got shape={x.shape}."
            )

        expected_c = int(self.config.c_in)
        if x.shape[0] != expected_c:
            raise ValueError(
                f"Channel mismatch: model expects {expected_c}, "
                f"input sample has {x.shape[0]}."
            )

        if not np.isfinite(x).all():
            raise ValueError("Online input contains NaN or Inf.")

        return x

    def _score_window_raw(
        self,
        window_raw: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        self._ensure_runtime_ready()

        window_raw = np.asarray(window_raw, dtype=np.float32)
        expected_shape = (
            int(self.config.seq_len),
            int(self.config.c_in),
        )
        if window_raw.shape != expected_shape:
            raise ValueError(
                f"Expected window {expected_shape}, got {window_raw.shape}."
            )

        # Reuse scaler fitted only on historical normal training data.
        window_scaled = self.scaler.transform(
            window_raw
        ).astype(np.float32, copy=False)

        batch_x = torch.from_numpy(window_scaled).unsqueeze(0)
        batch_x = batch_x.to(self.device, non_blocking=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        model_t0 = time.perf_counter()

        with torch.inference_mode():
            with self._controlled_inference_rng():
                outputs, _, _ = self.model(batch_x)

                temp_score = torch.mean(
                    (batch_x - outputs) ** 2,
                    dim=-1,
                )

                freq_score = torch.mean(
                    self._online_freq_criterion(
                        batch_x,
                        outputs,
                    ),
                    dim=-1,
                )

                point_scores = (
                    temp_score
                    + float(self.config.score_lambda) * freq_score
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        model_inference_ms = (
            time.perf_counter() - model_t0
        ) * 1000.0

        scores = (
            point_scores.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

        if scores.shape != (int(self.config.seq_len),):
            raise RuntimeError(
                f"Expected {self.config.seq_len} point scores, "
                f"got {scores.shape}."
            )

        if not np.isfinite(scores).all():
            raise RuntimeError(
                "CATCH produced NaN/Inf point scores. "
                "Check inference patch geometry and model stability."
            )

        return scores, float(model_inference_ms)

    # ------------------------------------------------------------------
    # Online push
    # ------------------------------------------------------------------
    def online_push(
        self,
        sample: Sequence[float],
        timestamp: Any = None,
    ) -> Optional[OnlineScoreBatch]:
        self._ensure_runtime_ready()
        x = self._validate_sample(sample)

        self._online_buffer.append(x)
        self._online_timestamp_buffer.append(timestamp)
        self._online_samples_seen += 1

        if self._online_samples_seen < self._online_next_infer_at:
            return None

        if self._online_samples_seen != self._online_next_infer_at:
            raise RuntimeError(
                "Online schedule lost synchronization: "
                f"samples_seen={self._online_samples_seen}, "
                f"next_infer_at={self._online_next_infer_at}."
            )

        if len(self._online_buffer) != int(self.config.seq_len):
            raise RuntimeError(
                f"Buffer length={len(self._online_buffer)}, "
                f"expected {self.config.seq_len}."
            )

        e2e_t0 = time.perf_counter()

        window_raw = np.stack(self._online_buffer, axis=0)
        point_scores, model_inference_ms = self._score_window_raw(window_raw)

        formal_scores = point_scores[
            self.emit_start_offset:
            self.emit_end_offset + 1
        ].copy()

        if len(formal_scores) != self.online_stride:
            raise RuntimeError(
                f"Formal score block has {len(formal_scores)} values; "
                f"expected {self.online_stride}."
            )

        window_end = self._online_samples_seen
        window_start = window_end - int(self.config.seq_len) + 1

        score_start = window_start + self.emit_start_offset
        score_end = window_start + self.emit_end_offset

        window_timestamps = list(self._online_timestamp_buffer)
        formal_timestamps = window_timestamps[
            self.emit_start_offset:
            self.emit_end_offset + 1
        ]
        if all(t is None for t in formal_timestamps):
            formal_timestamps = None

        labels = None
        alarm = None

        if self.online_threshold is not None:
            labels = (
                formal_scores > self.online_threshold
            ).astype(np.int8)
            alarm = bool(np.any(labels))

        trigger_end_to_end_ms = (
            time.perf_counter() - e2e_t0
        ) * 1000.0

        result = OnlineScoreBatch(
            window_start_index=int(window_start),
            window_end_index=int(window_end),
            score_start_index=int(score_start),
            score_end_index=int(score_end),
            scores=formal_scores,
            latest_score=float(formal_scores[-1]),
            timestamps=formal_timestamps,
            labels=labels,
            alarm=alarm,
            model_inference_ms=float(model_inference_ms),
            trigger_end_to_end_ms=float(trigger_end_to_end_ms),
        )

        self._online_next_infer_at += self.online_stride
        return result

    # ------------------------------------------------------------------
    # Offline replay of exact online schedule
    # ------------------------------------------------------------------
    def replay_online(
        self,
        data,
        timestamps: Optional[Iterable[Any]] = None,
        reset_before: bool = True,
    ) -> list[OnlineScoreBatch]:
        self._ensure_runtime_ready()

        if isinstance(data, pd.DataFrame):
            x = data.values.astype(np.float32, copy=False)
        else:
            x = np.asarray(data, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"data must be [T, C], got {x.shape}.")

        if x.shape[1] != int(self.config.c_in):
            raise ValueError(
                f"Expected {self.config.c_in} features, got {x.shape[1]}."
            )

        if not np.isfinite(x).all():
            raise ValueError("Replay data contains NaN or Inf.")

        if timestamps is None:
            ts_list = [None] * len(x)
        else:
            ts_list = list(timestamps)
            if len(ts_list) != len(x):
                raise ValueError(
                    "timestamps length must equal data length."
                )

        if reset_before:
            self.reset_online()

        results: list[OnlineScoreBatch] = []
        for row, ts in zip(x, ts_list):
            out = self.online_push(row, timestamp=ts)
            if out is not None:
                results.append(out)

        return results

    # ------------------------------------------------------------------
    # Online-safe threshold
    # ------------------------------------------------------------------
    def calibrate_online_threshold(
        self,
        normal_calibration_data,
        anomaly_ratio_percent: float = 1.0,
    ) -> float:
        ratio = float(anomaly_ratio_percent)

        if not (0.0 < ratio < 100.0):
            raise ValueError(
                "anomaly_ratio_percent must be in (0, 100)."
            )

        results = self.replay_online(
            normal_calibration_data,
            reset_before=True,
        )

        if not results:
            raise ValueError(
                "Calibration data produced no formal online score. "
                f"Need at least {self.config.seq_len} samples."
            )

        scores = np.concatenate(
            [r.scores for r in results],
            axis=0,
        )

        threshold = float(
            np.percentile(
                scores,
                100.0 - ratio,
            )
        )

        self.online_threshold = threshold
        self.reset_online()
        return threshold

    # ------------------------------------------------------------------
    # Train once
    # ------------------------------------------------------------------
    def fit_for_online(
        self,
        normal_data: pd.DataFrame,
        calibration_ratio: float = 0.20,
        anomaly_ratio_percent: float = 1.0,
    ) -> float:
        if not isinstance(normal_data, pd.DataFrame):
            raise TypeError(
                "normal_data must be a pandas DataFrame."
            )

        if normal_data.isna().any().any():
            raise ValueError(
                "normal_data contains NaN."
            )

        ratio = float(calibration_ratio)
        if not (0.05 <= ratio <= 0.50):
            raise ValueError(
                "calibration_ratio must be between 0.05 and 0.50."
            )

        split = int(
            len(normal_data) * (1.0 - ratio)
        )

        fit_data = normal_data.iloc[:split].copy()
        calibration_data = normal_data.iloc[split:].copy()

        if len(calibration_data) < int(self.config.seq_len):
            required_total = int(
                np.ceil(
                    int(self.config.seq_len) / ratio
                )
            )
            raise ValueError(
                f"Calibration section has only "
                f"{len(calibration_data)} rows, "
                f"but seq_len={self.config.seq_len}. "
                f"With calibration_ratio={ratio}, "
                f"use at least about {required_total} total rows."
            )

        # Preserve the existing guard for the author's training loop.
        internal_train_rows = int(len(fit_data) * 0.8)
        train_windows = (
            internal_train_rows
            - int(self.config.seq_len)
            + 1
        )
        if train_windows <= 0:
            raise ValueError(
                "Training part is too short to form one CATCH window."
            )

        approx_loader_batches = int(
            np.ceil(
                train_windows / int(self.config.batch_size)
            )
        )
        if approx_loader_batches < 10:
            raise ValueError(
                "Training data is too short for the author's current "
                "detect_fit() loop with this batch size. "
                f"Estimated train batches={approx_loader_batches}; "
                "need at least 10. Use more normal data or a smaller "
                "training batch size."
            )

        self.feature_names = [
            str(c) for c in normal_data.columns
        ]

        # Keep original CATCH training implementation.
        self.detect_fit(
            fit_data,
            fit_data,
        )

        # Restore EarlyStopping's best weights exactly as before.
        self.use_best_trained_weights()

        threshold = self.calibrate_online_threshold(
            calibration_data,
            anomaly_ratio_percent=anomaly_ratio_percent,
        )

        return threshold

    # ------------------------------------------------------------------
    # Repeatability diagnostic
    # ------------------------------------------------------------------
    def repeatability_check(
        self,
        raw_window: np.ndarray,
        repeats: int = 3,
    ) -> dict:
        repeats = int(repeats)
        if repeats < 2:
            raise ValueError("repeats must be >= 2.")

        all_scores = []
        for _ in range(repeats):
            s, _ = self._score_window_raw(raw_window)
            all_scores.append(s)

        base = all_scores[0]
        max_abs_diff = max(
            float(np.max(np.abs(s - base)))
            for s in all_scores[1:]
        )

        return {
            "repeats": repeats,
            "max_abs_diff": max_abs_diff,
            "mask_seed": self.inference_mask_seed,
        }

    # ------------------------------------------------------------------
    # Persistent bundle
    # ------------------------------------------------------------------
    def save_online_bundle(self, path) -> Path:
        self._ensure_runtime_ready()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        scaler_state = {
            "mean_": np.asarray(self.scaler.mean_, dtype=np.float64),
            "scale_": np.asarray(self.scaler.scale_, dtype=np.float64),
            "var_": np.asarray(self.scaler.var_, dtype=np.float64),
            "n_features_in_": int(self.scaler.n_features_in_),
            "n_samples_seen_": np.asarray(self.scaler.n_samples_seen_),
        }

        bundle = {
            "bundle_version": self.BUNDLE_VERSION,
            "config": dict(vars(self.config)),
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in self.model.state_dict().items()
            },
            "scaler_state": scaler_state,
            "feature_names": self.feature_names,
            "online_stride": int(self.online_stride),
            "online_threshold": self.online_threshold,
            "inference_mask_seed": self.inference_mask_seed,
            "right_context": int(self.right_context),
            "emit_start_offset": int(self.emit_start_offset),
            "emit_end_offset": int(self.emit_end_offset),
        }

        torch.save(bundle, path)
        return path

    @classmethod
    def load_online_bundle(
        cls,
        path,
        device: str = "auto",
    ) -> "CATCHOnline":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        try:
            bundle = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            bundle = torch.load(path, map_location="cpu")

        version = bundle.get("bundle_version")
        if version != cls.BUNDLE_VERSION:
            raise ValueError(
                f"Bundle version={version} is not compatible "
                f"with online code version={cls.BUNDLE_VERSION}. "
                "Please retrain/recreate the online bundle."
            )

        config_dict = dict(bundle["config"])

        online = cls(
            online_stride=int(bundle["online_stride"]),
            online_threshold=bundle.get("online_threshold"),
            inference_mask_seed=bundle.get(
                "inference_mask_seed",
                2025,
            ),
            **config_dict,
        )

        if device == "auto":
            online.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            online.device = torch.device(device)

        if (
            online.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )

        if not hasattr(online.config, "task_name"):
            online.config.task_name = "anomaly_detection"

        online.model = CATCHModel(online.config)
        online.model.load_state_dict(bundle["model_state_dict"])

        scaler_state = bundle["scaler_state"]
        online.scaler.mean_ = np.asarray(
            scaler_state["mean_"], dtype=np.float64
        )
        online.scaler.scale_ = np.asarray(
            scaler_state["scale_"], dtype=np.float64
        )
        online.scaler.var_ = np.asarray(
            scaler_state["var_"], dtype=np.float64
        )
        online.scaler.n_features_in_ = int(
            scaler_state["n_features_in_"]
        )
        online.scaler.n_samples_seen_ = np.asarray(
            scaler_state["n_samples_seen_"]
        )

        online.feature_names = bundle.get("feature_names")

        # Recomputed geometry must match what was saved.
        for key in (
            "right_context",
            "emit_start_offset",
            "emit_end_offset",
        ):
            if key in bundle:
                actual = int(getattr(online, key))
                saved = int(bundle[key])
                if actual != saved:
                    raise RuntimeError(
                        f"Online geometry mismatch for {key}: "
                        f"bundle={saved}, recomputed={actual}."
                    )

        online._online_runtime_ready = False
        online.prepare_online_runtime()
        online.reset_online()

        return online
