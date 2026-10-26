#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss_imputer.diffusion import MixedDiffusion  # noqa: E402
from dss_imputer.generative_metrics import (  # noqa: E402
    METRIC_PROTOCOL_VERSION,
    aggregate_metric_trials,
    evaluate_generative_quality,
    generative_metric_seed,
    evaluate_masked_numerical_accuracy,
)
from dss_imputer.inference import impute_dataframe, load_imputer  # noqa: E402
from dss_imputer.masks import (  # noqa: E402
    MaskDistribution,
    sample_artificial_mask,
)
from dss_imputer.model import MaskConditionedDenoiser  # noqa: E402
from dss_imputer.schema import EncodedTable, TablePreprocessor  # noqa: E402
from scripts.training_data import add_dataset_arguments, resolve_dataset_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one mask-conditioned diffusion imputer for a table."
    )
    add_dataset_arguments(parser)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument(
        "--validation-csv",
        type=Path,
        help="Separate table used for validation and best checkpoint selection.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="Continue from saved weights and preprocessing with a fresh optimizer/scheduler; epochs are additional.",
    )
    parser.add_argument(
        "--artifact-suffix", default="",
        help="Suffix for all outputs, e.g. continue -> last_continue.pt; defaults to continue when resuming.",
    )
    parser.add_argument(
        "--continuous-cols",
        nargs="+",
        default=["iops", "lat", "n_jobs", "iodepth"],
        help="Columns modeled with numerical Gaussian diffusion (lat is latency).",
    )
    parser.add_argument(
        "--log-continuous-cols",
        nargs="*",
        default=[],
        help="Continuous columns encoded as log(x + epsilon) before standardization.",
    )
    parser.add_argument("--log-epsilon", type=float, default=1e-6)
    parser.add_argument("--exclude-cols", nargs="*", default=["id"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save an epoch_XXXX.pt snapshot every N epochs; 0 disables snapshots.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--group-column",
        default="id",
        help="Reject overlapping groups between the training and validation files.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after this many stale epochs; 0 disables early stopping.",
    )
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--categorical-loss-weight", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument(
        "--architecture",
        choices=["transformer", "hybrid", "mlp"],
        default="hybrid",
        help=(
            "Denoiser backbone: transformer, hybrid Transformer -> MLP -> "
            "Transformer, or a flattened token MLP."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help=(
            "Transformer layers, layers in each hybrid Transformer, or hidden "
            "layers for the MLP backbone."
        ),
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--mlp-width",
        type=int,
        default=512,
        help="Hidden MLP width for the hybrid and MLP architectures.",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--single-probability", type=float, default=0.00)
    parser.add_argument("--mcar-probability", type=float, default=1.00)
    parser.add_argument("--loco-probability", type=float, default=0.00)
    parser.add_argument("--semantic-probability", type=float, default=0.00)
    parser.add_argument("--stress-probability", type=float, default=0.00)
    parser.add_argument(
        "--min-mask-rate",
        type=float,
        default=0.30,
        help="Minimum per-row MCAR target probability (default: 0.30).",
    )
    parser.add_argument(
        "--max-mask-rate",
        type=float,
        default=0.30,
        help="Maximum per-row MCAR target probability (default: 0.30).",
    )
    parser.add_argument("--stress-min-rate", type=float, default=0.50)
    parser.add_argument("--stress-max-rate", type=float, default=0.80)
    parser.add_argument("--min-context", type=int, default=1)
    parser.add_argument(
        "--semantic-group",
        action="append",
        default=None,
        metavar="NAME=COL1,COL2",
        help="Optional repeatable mask group. Disabled unless explicitly configured.",
    )
    parser.add_argument(
        "--training-loco-col",
        default=None,
        help="Restrict LOCO training masks to one modeled column, e.g. iops or lat.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--generative-validation-interval",
        type=int,
        default=5,
        help="Run stochastic distribution-quality validation every N epochs; 0 disables it.",
    )
    parser.add_argument(
        "--generative-validation-rows",
        type=int,
        default=2048,
        help="Fixed validation rows used for stochastic reconstruction; 0 uses all rows.",
    )
    parser.add_argument(
        "--generative-validation-rate",
        type=float,
        default=0.30,
        help="Fraction of observed cells hidden in the fixed validation scenario.",
    )
    parser.add_argument(
        "--generative-validation-loco-cols",
        nargs="*",
        default=["iops", "lat"],
        help="Additional fixed LOCO reconstruction scenarios.",
    )
    parser.add_argument("--generative-validation-batch-size", type=int, default=1024)
    parser.add_argument(
        "--generative-validation-samples",
        type=int,
        default=1,
        help="Independent stochastic reconstructions per validation scenario.",
    )
    parser.add_argument("--generative-dcr-rows", type=int, default=500)
    parser.add_argument(
        "--clearml-project",
        default=None,
        help="If set, report losses and generative metrics to ClearML.",
    )
    parser.add_argument("--clearml-task-name", default=None)
    parser.add_argument("--clearml-tags", nargs="*", default=[])
    args = resolve_dataset_arguments(parser.parse_args(), parser, "train_csv", "validation_csv")
    if args.train_csv is None or args.validation_csv is None:
        parser.error("Supply --train-csv and --validation-csv, or --dataset-config and --dataset.")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_clearml(args: argparse.Namespace, config: dict[str, Any]):
    if not args.clearml_project:
        return None
    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError(
            "ClearML logging was requested, but clearml is not installed in the "
            "training environment."
        ) from exc

    task_name = args.clearml_task_name or f"dss_imputer_{args.output_dir.name}"
    task = Task.init(project_name=args.clearml_project, task_name=task_name)
    if args.clearml_tags:
        task.add_tags(args.clearml_tags)
    task.connect(config, name="dss_imputer_config")
    return task


def report_clearml(task, epoch: int, values: dict[str, float | int]) -> None:
    if task is None:
        return
    logger = task.get_logger()
    for name, value in values.items():
        if name == "epoch" or not isinstance(value, (int, float)):
            continue
        if not np.isfinite(value):
            continue
        if name.startswith("generation_"):
            title = "Generative validation"
        elif "loss" in name:
            title = "Loss"
        elif name.startswith("train_mask_") or name.startswith("validation_mask_"):
            title = "Mask mixture"
        else:
            title = "Optimization"
        logger.report_scalar(
            title=title, series=name, value=float(value), iteration=epoch
        )


def subset(encoded: EncodedTable, indices: np.ndarray) -> EncodedTable:
    return EncodedTable(
        numerical=encoded.numerical[indices],
        categorical=encoded.categorical[indices],
        numerical_available=encoded.numerical_available[indices],
        categorical_available=encoded.categorical_available[indices],
    )


def save_periodic_checkpoint(
    output_dir: Path, epoch: int, interval: int, state: dict[str, object], suffix: str = ""
) -> Path | None:
    if interval == 0 or epoch % interval:
        return None
    path = output_dir / f"epoch_{epoch:04d}{'_' + suffix if suffix else ''}.pt"
    torch.save(state, path)
    return path


def artifact_path(output_dir: Path, filename: str, suffix: str) -> Path:
    path = Path(filename)
    return output_dir / f"{path.stem}{'_' + suffix if suffix else ''}{path.suffix}"


def resolve_semantic_groups(
    specifications: list[str], preprocessor: TablePreprocessor
) -> tuple[dict[str, list[str]], list[list[int]]]:
    model_columns = preprocessor.model_columns
    original_columns = set(preprocessor.original_columns)
    resolved_names: dict[str, list[str]] = {}
    resolved_indices: list[list[int]] = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Invalid semantic group {specification!r}; expected NAME=COL1,COL2."
            )
        name, raw_columns = specification.split("=", 1)
        columns = [
            column.strip() for column in raw_columns.split(",") if column.strip()
        ]
        unknown = sorted(set(columns) - original_columns)
        if unknown:
            raise ValueError(f"Semantic group {name!r} has unknown columns: {unknown}")
        modeled = [column for column in columns if column in model_columns]
        indices = list(dict.fromkeys(model_columns.index(column) for column in modeled))
        if not indices:
            print(
                f"Skipping semantic group {name!r}: all its columns are constant "
                "or excluded."
            )
            continue
        resolved_names[name.strip()] = [model_columns[index] for index in indices]
        resolved_indices.append(indices)
    return resolved_names, resolved_indices


def make_loader(
    encoded: EncodedTable,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(encoded.numerical),
        torch.from_numpy(encoded.categorical),
        torch.from_numpy(encoded.numerical_available),
        torch.from_numpy(encoded.categorical_available),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def prepare_generative_validation(
    frame: pd.DataFrame,
    encoded: EncodedTable,
    preprocessor: TablePreprocessor,
    rows: int,
    mask_rate: float,
    loco_columns: list[str],
    min_context: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, np.ndarray]]]:
    if not 0 < mask_rate < 1:
        raise ValueError("generative-validation-rate must be between zero and one.")
    if rows < 0:
        raise ValueError("generative-validation-rows cannot be negative.")

    count = len(frame) if rows == 0 else min(rows, len(frame))
    generator = np.random.default_rng(seed)
    selected_indices = generator.choice(len(frame), size=count, replace=False)
    selected_encoded = subset(encoded, selected_indices)
    unknown_loco = sorted(set(loco_columns) - set(preprocessor.model_columns))
    if unknown_loco:
        raise ValueError(
            "Generative validation LOCO columns are not modeled: " f"{unknown_loco}"
        )
    distribution = MaskDistribution(
        single_probability=0.0,
        mcar_probability=1.0,
        loco_probability=0.0,
        semantic_probability=0.0,
        stress_probability=0.0,
        min_rate=mask_rate,
        max_rate=mask_rate,
        min_context=min_context,
    )
    artificial_mask, _ = sample_artificial_mask(
        torch.from_numpy(selected_encoded.available),
        distribution,
        generator=torch.Generator().manual_seed(seed),
        mode="mcar",
    )
    real = frame.iloc[selected_indices].reset_index(drop=True)
    available = selected_encoded.available
    scenarios: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    def add_scenario(name: str, mask: np.ndarray) -> None:
        masked = real.copy()
        for column_index, column in enumerate(preprocessor.model_columns):
            masked[column] = masked[column].mask(mask[:, column_index])
        scenarios[name] = (masked, mask)

    add_scenario(f"mcar_{round(mask_rate * 100):02d}", artificial_mask.numpy())
    valid_rows = available.sum(axis=1) > min_context
    for column in loco_columns:
        column_index = preprocessor.model_columns.index(column)
        mask = np.zeros_like(available, dtype=bool)
        mask[:, column_index] = available[:, column_index] & valid_rows
        if not mask[:, column_index].any():
            raise ValueError(f"LOCO validation has no eligible values for {column!r}.")
        add_scenario(f"loco_{column}", mask)

    return real, scenarios


def format_generative_metrics(metrics: dict[str, float]) -> str:
    order = [
        "target_overall_quality",
        "target_column_shapes",
        "target_column_pair_trends",
        "target_c2st",
        "target_alpha_precision",
        "target_beta_recall",
        "overall_quality",
        "column_shapes",
        "column_pair_trends",
        "c2st",
        "alpha_precision",
        "beta_recall",
        "authenticity",
        "dcr",
    ]
    distribution_metrics = " ".join(
        (
            f"{name}={metrics[name]:.4f}±{metrics[f'{name}_std']:.4f}"
            if f"{name}_std" in metrics
            else f"{name}={metrics[name]:.4f}"
        )
        for name in order
        if name in metrics
    )
    accuracy_metrics: list[str] = []
    for column in ["iops", "lat"]:
        if f"{column}_r2" in metrics:
            accuracy_metrics.extend(
                [
                    f"{column}_r2={metrics[f'{column}_r2']:.4f}",
                    f"{column}_mse={metrics[f'{column}_mse']:.6g}",
                    f"{column}_rmse={metrics[f'{column}_rmse']:.6g}",
                ]
            )
    return " ".join([distribution_metrics, *accuracy_metrics])


def run_epoch(
    diffusion: MixedDiffusion,
    loader: DataLoader,
    mask_distribution: MaskDistribution,
    semantic_groups: list[list[int]],
    loco_indices: list[int] | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> dict[str, float]:
    training = optimizer is not None
    diffusion.train(training)
    generator = torch.Generator().manual_seed(seed)
    totals = {"loss": 0.0, "numerical_loss": 0.0, "categorical_loss": 0.0}
    mode_counts = {
        "single": 0,
        "mcar": 0,
        "loco": 0,
        "semantic": 0,
        "stress": 0,
    }
    rows = 0
    batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for (
            numerical,
            categorical,
            numerical_available,
            categorical_available,
        ) in loader:
            available = torch.cat([numerical_available, categorical_available], dim=1)
            artificial, mask_mode = sample_artificial_mask(
                available,
                mask_distribution,
                generator=generator,
                semantic_groups=semantic_groups,
                loco_indices=loco_indices,
            )
            numerical_targets = artificial[:, : numerical.shape[1]]
            categorical_targets = artificial[:, numerical.shape[1] :]
            mode_counts[mask_mode] += 1
            batches += 1

            numerical = numerical.to(device, non_blocking=True)
            categorical = categorical.to(device, non_blocking=True)
            numerical_available = numerical_available.to(device, non_blocking=True)
            categorical_available = categorical_available.to(device, non_blocking=True)
            numerical_targets = numerical_targets.to(device, non_blocking=True)
            categorical_targets = categorical_targets.to(device, non_blocking=True)

            losses = diffusion.training_loss(
                numerical,
                categorical,
                numerical_available,
                categorical_available,
                numerical_targets,
                categorical_targets,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
                optimizer.step()

            batch_rows = numerical.shape[0]
            rows += batch_rows
            for name in totals:
                totals[name] += float(losses[name].detach()) * batch_rows

    metrics = {name: value / max(rows, 1) for name, value in totals.items()}
    metrics.update(
        {
            f"mask_{mode}_fraction": count / max(batches, 1)
            for mode, count in mode_counts.items()
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.resume_checkpoint and not args.artifact_suffix:
        args.artifact_suffix = "continue"
    if args.artifact_suffix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.artifact_suffix):
        raise ValueError("artifact-suffix must contain only letters, digits, underscores, or hyphens.")
    if args.resume_checkpoint:
        for filename in ("last.pt", "best.pt", "history.csv", "run_config.json"):
            destination = artifact_path(args.output_dir, filename, args.artifact_suffix)
            if destination.exists():
                raise ValueError(f"Continuation output already exists: {destination}. Choose a fresh artifact-suffix.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval cannot be negative.")
    if args.patience < 0:
        raise ValueError("patience cannot be negative; use 0 to disable early stopping.")
    if args.generative_validation_interval < 0:
        raise ValueError("generative-validation-interval cannot be negative.")
    if args.generative_validation_batch_size < 1:
        raise ValueError("generative-validation-batch-size must be positive.")
    if args.generative_validation_samples < 1:
        raise ValueError("generative-validation-samples must be positive.")
    if args.generative_dcr_rows < 2:
        raise ValueError("generative-dcr-rows must be at least two.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    frame = pd.read_csv(args.train_csv, low_memory=False, na_values=["?"])
    if args.train_csv.resolve() == args.validation_csv.resolve():
        raise ValueError("Training and validation CSVs must be different files.")
    validation_frame = pd.read_csv(
        args.validation_csv, low_memory=False, na_values=["?"]
    )
    if args.max_rows is not None:
        frame = frame.sample(
            n=min(args.max_rows, len(frame)), random_state=args.seed
        ).reset_index(drop=True)
    if frame.empty or validation_frame.empty:
        raise ValueError("Training and validation CSVs must both contain rows.")
    missing_columns = sorted(set(frame.columns) - set(validation_frame.columns))
    if missing_columns:
        raise ValueError(f"Validation CSV is missing columns: {missing_columns}")
    if args.group_column and args.group_column in frame and args.group_column in validation_frame:
        train_groups = set(frame[args.group_column].dropna().astype(str))
        validation_groups = set(validation_frame[args.group_column].dropna().astype(str))
        overlap = train_groups & validation_groups
        if overlap:
            raise ValueError(
                f"Training and validation share {len(overlap)} {args.group_column!r} groups."
            )
    split_method = "external_csv"
    source_checkpoint: dict[str, Any] | None = None
    if args.resume_checkpoint:
        diffusion, preprocessor, source_checkpoint = load_imputer(args.resume_checkpoint, device)
        denoiser = diffusion.denoiser
        # The source vocabulary, normalization, and architecture must not be refit.
        args.continuous_cols = preprocessor.numerical_columns
        args.log_continuous_cols = preprocessor.log_continuous_columns
        args.log_epsilon = preprocessor.log_epsilon
        args.exclude_cols = preprocessor.excluded_columns
        args.architecture = denoiser.architecture
        args.d_model, args.layers, args.heads = denoiser.d_model, denoiser.layers, denoiser.heads
        args.mlp_width, args.dropout = denoiser.mlp_width, denoiser.dropout
        args.ff_multiplier = denoiser.ff_multiplier
        args.diffusion_steps = diffusion.steps
        args.categorical_loss_weight = diffusion.categorical_loss_weight
    else:
        preprocessor = TablePreprocessor(
            continuous_columns=args.continuous_cols,
            excluded_columns=args.exclude_cols,
            log_continuous_columns=args.log_continuous_cols,
            log_epsilon=args.log_epsilon,
        ).fit(frame)
    start_epoch = int(source_checkpoint.get("epoch", 0)) if source_checkpoint else 0
    final_epoch = start_epoch + args.epochs
    encoded_train = preprocessor.transform(frame)
    encoded_validation = preprocessor.transform(validation_frame)
    semantic_group_specs = args.semantic_group or []
    semantic_group_names, semantic_group_indices = resolve_semantic_groups(
        semantic_group_specs, preprocessor
    )
    training_loco_indices: list[int] | None = None
    if args.training_loco_col is not None:
        if args.training_loco_col not in preprocessor.model_columns:
            raise ValueError(
                "training-loco-col is not a modeled non-constant column: "
                f"{args.training_loco_col!r}"
            )
        if args.loco_probability <= 0:
            raise ValueError(
                "training-loco-col requires a positive --loco-probability."
            )
        training_loco_indices = [
            preprocessor.model_columns.index(args.training_loco_col)
        ]
    train_loader = make_loader(
        encoded_train, args.batch_size, True, args.num_workers
    )
    validation_loader = make_loader(
        encoded_validation, args.batch_size, False, args.num_workers
    )

    generative_real: pd.DataFrame | None = None
    generative_scenarios: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    train_reference: pd.DataFrame | None = None
    if args.generative_validation_interval > 0:
        generative_real, generative_scenarios = prepare_generative_validation(
            validation_frame,
            encoded_validation,
            preprocessor,
            args.generative_validation_rows,
            args.generative_validation_rate,
            args.generative_validation_loco_cols,
            args.min_context,
            args.seed + 200_000,
        )
        train_reference = frame.loc[:, preprocessor.model_columns].reset_index(drop=True)

    if source_checkpoint is None:
        denoiser = MaskConditionedDenoiser(
            numerical_features=len(preprocessor.numerical_columns),
            categorical_sizes=preprocessor.categorical_sizes,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            ff_multiplier=args.ff_multiplier,
            architecture=args.architecture,
            mlp_width=args.mlp_width,
        )
        diffusion = MixedDiffusion(
            denoiser,
            steps=args.diffusion_steps,
            categorical_loss_weight=args.categorical_loss_weight,
        ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in diffusion.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    mask_distribution = MaskDistribution(
        single_probability=args.single_probability,
        mcar_probability=args.mcar_probability,
        loco_probability=args.loco_probability,
        semantic_probability=args.semantic_probability,
        stress_probability=args.stress_probability,
        min_rate=args.min_mask_rate,
        max_rate=args.max_mask_rate,
        stress_min_rate=args.stress_min_rate,
        stress_max_rate=args.stress_max_rate,
        min_context=args.min_context,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    def output_path(filename: str) -> Path:
        return artifact_path(args.output_dir, filename, args.artifact_suffix)

    preprocessor.save(output_path("preprocessor.json"))
    run_config = vars(args).copy()
    run_config["metric_protocol"] = METRIC_PROTOCOL_VERSION
    run_config["train_csv"] = str(args.train_csv)
    run_config["validation_csv"] = str(args.validation_csv)
    run_config["output_dir"] = str(args.output_dir)
    run_config["dataset_config"] = str(args.dataset_config) if args.dataset_config else None
    run_config["resume_checkpoint"] = str(args.resume_checkpoint) if args.resume_checkpoint else None
    run_config["start_epoch"] = start_epoch
    run_config["final_epoch"] = final_epoch
    run_config["continuation_mode"] = "weights_fresh_optimizer" if source_checkpoint else None
    run_config["device_resolved"] = str(device)
    run_config["split_method"] = split_method
    run_config["train_rows"] = len(frame)
    run_config["validation_rows"] = len(validation_frame)
    run_config["trainable_parameters"] = trainable_parameters
    run_config["semantic_groups_resolved"] = semantic_group_names
    run_config["training_loco_indices_resolved"] = training_loco_indices
    output_path("run_config.json").write_text(json.dumps(run_config, indent=2))
    if generative_real is not None:
        generative_real.to_csv(
            output_path("generative_validation_real.csv"), index=False
        )
        for scenario, (masked, mask) in generative_scenarios.items():
            masked.to_csv(
                output_path(f"generative_validation_{scenario}_masked.csv"),
                index=False,
            )
            pd.DataFrame(mask, columns=preprocessor.model_columns).to_csv(
                output_path(f"generative_validation_{scenario}_mask.csv"),
                index=False,
            )
    clearml_task = init_clearml(args, run_config)

    def checkpoint(epoch: int, validation_loss: float) -> dict[str, object]:
        return {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state": diffusion.state_dict(),
            "model_config": denoiser.config(),
            "diffusion_config": diffusion.config(),
            "preprocessor": preprocessor.to_dict(),
            "run_config": run_config,
        }

    print(f"Device: {device}")
    if source_checkpoint:
        print(f"Continuing weights from {args.resume_checkpoint}, epoch {start_epoch}; {args.epochs} additional epochs.")
        print(f"Fresh AdamW and cosine LR schedule: lr={args.learning_rate:g}, batch_size={args.batch_size}.")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Train/validation rows: {len(frame)}/{len(validation_frame)}")
    print(f"Validation source: {args.validation_csv} ({split_method})")
    print(
        f"WARNING: the validation CSV selects {output_path('best.pt').name} and cannot also serve "
        "as an independent final test."
    )
    print(f"Numerical columns: {preprocessor.numerical_columns}")
    print(
        "Log-transformed numerical columns: "
        f"{preprocessor.log_continuous_columns} (eps={preprocessor.log_epsilon:g})"
    )
    print(f"Categorical columns: {preprocessor.categorical_columns}")
    print(f"Constant columns: {list(preprocessor.constant_values)}")
    print(f"Excluded columns: {preprocessor.excluded_columns}")
    print(f"Semantic mask groups: {semantic_group_names}")
    print(f"Training LOCO column: {args.training_loco_col}")
    print(f"Mask probabilities: {mask_distribution.probabilities}")
    print(
        "Early stopping: "
        + ("disabled" if args.patience == 0 else f"patience={args.patience}")
    )
    if generative_real is not None:
        print(
            "Generative validation: "
            f"rows={len(generative_real)}, "
            f"scenarios={list(generative_scenarios)}, "
            f"samples={args.generative_validation_samples}, "
            f"every={args.generative_validation_interval} epoch(s)"
        )

    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    generative_history: list[dict[str, float | int | str]] = []
    generative_error_history: list[dict[str, int | str]] = []
    for epoch in range(start_epoch + 1, final_epoch + 1):
        train_metrics = run_epoch(
            diffusion,
            train_loader,
            mask_distribution,
            semantic_group_indices,
            training_loco_indices,
            device,
            optimizer,
            args.seed + epoch,
        )
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.seed + 100_000)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 100_000)
            validation_metrics = run_epoch(
                diffusion,
                validation_loader,
                mask_distribution,
                semantic_group_indices,
                training_loco_indices,
                device,
                None,
                args.seed + 100_000,
            )
        scheduler.step()
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        row["learning_rate"] = optimizer.param_groups[0]["lr"]

        run_generative_validation = (
            generative_real is not None
            and bool(generative_scenarios)
            and train_reference is not None
            and (
                epoch == start_epoch + 1
                or epoch % args.generative_validation_interval == 0
                or epoch == final_epoch
            )
        )
        generative_results: list[tuple[str, dict[str, float], list[str]]] = []
        if run_generative_validation:
            for scenario_index, (scenario, (masked, mask)) in enumerate(
                generative_scenarios.items()
            ):
                generated_samples = impute_dataframe(
                    masked,
                    diffusion,
                    preprocessor,
                    device,
                    samples=args.generative_validation_samples,
                    batch_size=args.generative_validation_batch_size,
                    seed=args.seed + 300_000 + scenario_index,
                )
                trial_metrics: list[dict[str, float]] = []
                errors: list[str] = []
                for trial_index, generated in enumerate(generated_samples):
                    generated.to_csv(
                        output_path(
                            f"generative_validation_{scenario}_latest_"
                            f"trial_{trial_index:03d}.csv"
                        ),
                        index=False,
                    )
                    if trial_index == 0:
                        generated.to_csv(
                            output_path(f"generative_validation_{scenario}_latest.csv"),
                            index=False,
                        )
                    current_metrics, current_errors = evaluate_generative_quality(
                        generative_real.loc[:, preprocessor.model_columns],
                        generated.loc[:, preprocessor.model_columns],
                        train_reference,
                        sample_size=len(generative_real),
                        dcr_sample_size=args.generative_dcr_rows,
                        seed=generative_metric_seed(args.seed + 400_000, scenario),
                        c2st_seed=generative_metric_seed(910_000, scenario),
                        numerical_columns=preprocessor.numerical_columns,
                        categorical_columns=preprocessor.categorical_columns,
                        target_mask=mask,
                    )
                    current_metrics.update(
                        evaluate_masked_numerical_accuracy(
                            generative_real,
                            generated,
                            mask,
                            preprocessor.model_columns,
                            preprocessor.numerical_columns,
                        )
                    )
                    trial_metrics.append(current_metrics)
                    errors.extend(
                        f"trial {trial_index}: {error}" for error in current_errors
                    )
                    generative_error_history.extend(
                        {"epoch": epoch, "scenario": scenario, "trial": trial_index, "error": error}
                        for error in current_errors
                    )
                metrics = aggregate_metric_trials(trial_metrics)
                generation_row: dict[str, float | int | str] = {
                    "epoch": epoch,
                    "scenario": scenario,
                    "masked_rate": float(mask.mean()),
                    "samples": args.generative_validation_samples,
                    "metric_protocol": METRIC_PROTOCOL_VERSION,
                }
                generation_row.update(metrics)
                generative_history.append(generation_row)
                row.update(
                    {
                        f"generation_{scenario}_{name}": value
                        for name, value in metrics.items()
                    }
                )
                generative_results.append((scenario, metrics, errors))
            pd.DataFrame(generative_history).to_csv(
                output_path("generative_history.csv"), index=False
            )
            pd.DataFrame(
                generative_error_history, columns=["epoch", "scenario", "trial", "error"]
            ).to_csv(output_path("generative_metric_errors.csv"), index=False)

        history.append(row)
        pd.DataFrame(history).to_csv(output_path("history.csv"), index=False)

        print(
            f"Epoch {epoch:03d}/{final_epoch}: "
            f"train={train_metrics['loss']:.6f} "
            f"val={validation_metrics['loss']:.6f} "
            f"num={validation_metrics['numerical_loss']:.6f} "
            f"cat={validation_metrics['categorical_loss']:.6f}"
        )
        if run_generative_validation:
            for scenario, metrics, errors in generative_results:
                print(
                    f"  Generative validation [{scenario}]: "
                    + format_generative_metrics(metrics)
                )
                for error in errors:
                    print(f"  Generative metric warning [{scenario}]: {error}")
        report_clearml(clearml_task, epoch, row)
        state = checkpoint(epoch, validation_metrics["loss"])
        torch.save(state, output_path("last.pt"))
        periodic_path = save_periodic_checkpoint(
            args.output_dir, epoch, args.checkpoint_interval, state, args.artifact_suffix
        )
        if periodic_path is not None:
            print(f"Periodic checkpoint: {periodic_path}", flush=True)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            stale_epochs = 0
            torch.save(state, output_path("best.pt"))
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} stale epochs.")
                break

    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Checkpoint: {output_path('best.pt')}")
    if clearml_task is not None:
        for artifact in [
            output_path("history.csv"),
            output_path("generative_history.csv"),
            output_path("best.pt"),
            output_path("last.pt"),
        ]:
            if artifact.exists():
                clearml_task.upload_artifact(
                    artifact.name, artifact_object=str(artifact)
                )
        clearml_task.close()


if __name__ == "__main__":
    main()
