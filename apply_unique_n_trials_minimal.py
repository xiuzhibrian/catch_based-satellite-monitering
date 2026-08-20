#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal patch for tune_catch_online_optuna.py

Only changes:
1) --n-trials means the number of NEW, DIFFERENT, VALID parameter combinations
   that actually reach the CATCH runner in each stage.
2) Duplicate combinations and statically invalid/pruned-before-runner
   combinations do not consume --n-trials.
3) A new unique combination that reaches the runner counts once even if
   the runner later fails/OOMs.

Everything else in the original tuning script is left unchanged.
"""

from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path


NEW_RUN_STAGE = r"""def run_stage(args, stage: str, current_params: dict) -> tuple[dict, dict]:
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

    # Existing combinations that have already reached the real experiment.
    # Old statically-invalid trials do not have "full_params", because
    # suggest_for_stage() prunes them before full_params is recorded.
    seen_combinations = set()
    for old_trial in study.trials:
        full = old_trial.user_attrs.get("full_params")
        if isinstance(full, dict):
            key = json.dumps(
                full,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            seen_combinations.add(key)

    print("\n" + "=" * 88)
    print(f"[STAGE] {stage}")
    print(f"[TRIALS TO ADD] {args.n_trials}")
    print(f"[OBJECTIVE] {args.objective}")
    print("=" * 88)

    unique_experiments = 0
    optuna_attempts = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal unique_experiments

        suggested = suggest_for_stage(
            trial,
            stage=stage,
            current=current_params,
        )

        params = dict(current_params)
        params.update(suggested)

        combo_key = json.dumps(
            params,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # A repeated parameter combination is NOT a new experiment and
        # therefore does not consume --n-trials.
        if combo_key in seen_combinations:
            trial.set_user_attr("duplicate_combination", True)
            raise optuna.TrialPruned(
                "Duplicate parameter combination; not counted in --n-trials."
            )

        # Count the combination immediately before launching the CATCH runner.
        # Therefore:
        # - static-invalid combinations pruned in suggest_for_stage(): not counted
        # - duplicates: not counted
        # - a unique combination that actually launches but later OOMs/fails:
        #   counted once, because it was genuinely experimented
        seen_combinations.add(combo_key)
        unique_experiments += 1

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

    # --n-trials now counts UNIQUE REAL EXPERIMENTS, not raw Optuna trials.
    # Run one Optuna attempt at a time until enough different combinations
    # have actually reached the CATCH runner.
    #
    # The guard only prevents an endless loop if a finite search space has
    # already been exhausted or the sampler keeps proposing duplicates.
    max_optuna_attempts = max(args.n_trials * 100, 1000)

    while unique_experiments < args.n_trials:
        if optuna_attempts >= max_optuna_attempts:
            raise RuntimeError(
                f"Could not obtain {args.n_trials} different parameter "
                f"combinations for stage '{stage}' after "
                f"{optuna_attempts} Optuna attempts. "
                "The remaining search space may be exhausted."
            )

        study.optimize(
            objective,
            n_trials=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )
        optuna_attempts += 1

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
"""


OLD_HELP = """    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of new Optuna trials added per stage.",
    )"""

NEW_HELP = """    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help=(
            "Number of different parameter combinations actually "
            "experimented per stage."
        ),
    )"""


def patch_file(path: Path) -> None:
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"[ERROR] File not found: {path}")

    original = path.read_text(encoding="utf-8")
    text = original

    pattern = re.compile(
        r'def run_stage\(args, stage: str, current_params: dict\) -> tuple\[dict, dict\]:.*?'
        r'(?=\n# ---------------------------------------------------------------------\n# Final validation / test)',
        re.DOTALL,
    )

    match = pattern.search(text)
    if not match:
        raise SystemExit(
            "[ERROR] Could not locate the original run_stage() function."
        )

    text = text[:match.start()] + NEW_RUN_STAGE.rstrip() + "\n" + text[match.end():]

    if OLD_HELP not in text:
        raise SystemExit(
            "[ERROR] Could not locate the original --n-trials argument block."
        )
    text = text.replace(OLD_HELP, NEW_HELP, 1)

    backup = path.with_suffix(path.suffix + ".before_unique_ntrials.bak")
    shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")

    try:
        py_compile.compile(str(path), doraise=True)
    except Exception:
        shutil.copy2(backup, path)
        raise

    print(f"[OK] Modified : {path}")
    print(f"[OK] Backup   : {backup}")
    print("[OK] Syntax check passed.")
    print(
        "[SEMANTICS] --n-trials N = N different valid parameter "
        "combinations actually launched."
    )
    print(
        "[UNCHANGED] Search spaces, parameter values, pruning constraints, "
        "runner, metrics, output files, final evaluation."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        default="./scripts/tune_catch_online_optuna.py",
        help="Path to the original tune_catch_online_optuna.py",
    )
    args = parser.parse_args()
    patch_file(Path(args.target))


if __name__ == "__main__":
    main()
