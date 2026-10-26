from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from dss_imputer.diffusion import (
    MixedDiffusion,
    masked_cross_entropy,
    masked_mse,
)
from dss_imputer.generative_metrics import (
    aggregate_metric_trials,
    evaluate_generative_quality,
    evaluate_masked_numerical_accuracy,
    logistic_detection_quality,
)
from dss_imputer.masks import MaskDistribution, sample_artificial_mask
from dss_imputer.model import MaskConditionedDenoiser
from dss_imputer.schema import TablePreprocessor
from scripts.eval_dss_imputer import context_integrity_records, select_training_subset
from scripts.eval_dss_imputer import aggregate_deterministic_trials
from scripts.train_dss_imputer import save_periodic_checkpoint
from dss_imputer.splits import split_indices


class RecordingDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.categorical_sizes = [3, 2]
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        time: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del time, condition_mask
        self.inputs.append((numerical.detach().clone(), categorical.detach().clone()))
        noise = torch.zeros_like(numerical) + self.anchor
        logits = [
            torch.zeros(
                (len(categorical), size),
                device=categorical.device,
                dtype=numerical.dtype,
            )
            + self.anchor
            for size in self.categorical_sizes
        ]
        return noise, logits


class DssImputerTest(unittest.TestCase):
    def test_trial_metric_aggregation_reports_mean_and_sample_std(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "scenario": "loco_iops", "pattern": "loco", "rate": np.nan,
                    "column": "iops", "type": "numerical", "trial": 0,
                    "r2": 0.2, "mse": 4.0, "rmse": 2.0, "mae": 1.0,
                    "accuracy": np.nan, "macro_f1": np.nan,
                },
                {
                    "scenario": "loco_iops", "pattern": "loco", "rate": np.nan,
                    "column": "iops", "type": "numerical", "trial": 1,
                    "r2": 0.6, "mse": 2.0, "rmse": 1.0, "mae": 0.5,
                    "accuracy": np.nan, "macro_f1": np.nan,
                },
            ]
        )
        result = aggregate_deterministic_trials(frame).iloc[0]
        self.assertEqual(result["trials"], 2)
        self.assertAlmostEqual(result["r2_mean"], 0.4)
        self.assertAlmostEqual(result["r2_std"], np.sqrt(0.08))

    def test_periodic_checkpoints_are_saved_on_interval_boundaries(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            state = {"epoch": 99, "model_state": {"weight": torch.tensor([1.0])}}
            self.assertIsNone(save_periodic_checkpoint(output_dir, 99, 100, state))
            self.assertFalse((output_dir / "epoch_0099.pt").exists())

            for epoch in (100, 200):
                state["epoch"] = epoch
                path = save_periodic_checkpoint(output_dir, epoch, 100, state)
                self.assertEqual(path, output_dir / f"epoch_{epoch:04d}.pt")
                self.assertEqual(
                    torch.load(path, map_location="cpu", weights_only=False)["epoch"],
                    epoch,
                )

            self.assertIsNone(save_periodic_checkpoint(output_dir, 300, 0, state))
            self.assertEqual(len(list(output_dir.glob("epoch_*.pt"))), 2)

    def test_eval_reproduces_grouped_fit_and_validation_split(self) -> None:
        frame = pd.DataFrame(
            {"id": [f"group-{i // 2}" for i in range(20)], "value": range(20)}
        )
        with TemporaryDirectory() as directory:
            csv = Path(directory) / "train.csv"
            frame.to_csv(csv, index=False)
            checkpoint = {
                "run_config": {
                    "train_csv": str(csv),
                    "max_rows": None,
                    "validation_fraction": 0.2,
                    "group_column": "id",
                    "seed": 42,
                }
            }
            fit, method = select_training_subset(frame, checkpoint, csv, "fit")
            validation, validation_method = select_training_subset(
                frame, checkpoint, csv, "validation"
            )
            expected_fit, expected_validation, _ = split_indices(
                frame, 0.2, "id", 42
            )
            self.assertEqual(method, "group:id")
            self.assertEqual(validation_method, method)
            self.assertEqual(fit.value.tolist(), frame.iloc[expected_fit].value.tolist())
            self.assertEqual(
                validation.value.tolist(),
                frame.iloc[expected_validation].value.tolist(),
            )
            self.assertFalse(set(fit.id) & set(validation.id))
            with self.assertRaisesRegex(ValueError, "original train CSV"):
                select_training_subset(frame, checkpoint, Path(directory) / "test.csv", "fit")

    def test_metric_trials_report_mean_and_sample_std(self) -> None:
        metrics = aggregate_metric_trials(
            [{"c2st": 0.6, "beta_recall": 0.2}, {"c2st": 0.8, "beta_recall": 0.4}]
        )
        self.assertAlmostEqual(metrics["c2st"], 0.7)
        self.assertAlmostEqual(metrics["c2st_std"], np.sqrt(0.02))
        self.assertAlmostEqual(metrics["beta_recall"], 0.3)

    def test_c2st_uses_reproducible_cross_validation_folds(self) -> None:
        real = pd.DataFrame({"value": np.linspace(0.0, 1.0, 30)})
        generated = pd.DataFrame({"value": np.linspace(0.1, 1.1, 30)})
        metadata = {
            "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
            "columns": {"value": {"sdtype": "numerical"}},
        }
        first = logistic_detection_quality(real, generated, metadata, seed=17)
        np.random.seed(999)
        second = logistic_detection_quality(real, generated, metadata, seed=17)
        self.assertEqual(first, second)

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.frame = pd.DataFrame(
            {
                "iops": [10, 20, 30, np.nan, 50, 60],
                "lat": [2.0, 3.0, 4.0, 5.0, np.nan, 7.0],
                "iodepth": [1, 2, 4, 1, 2, 4],
                "io_type": ["read", "write", "read", "write", "read", None],
                "constant": ["ssd"] * 6,
                "id": [f"row-{index}" for index in range(6)],
            }
        )
        self.preprocessor = TablePreprocessor(["iops", "lat"], ["id"]).fit(self.frame)

    def test_schema_preserves_missingness_and_constants(self) -> None:
        encoded = self.preprocessor.transform(self.frame)
        self.assertEqual(
            set(self.preprocessor.model_columns),
            {"iops", "lat", "iodepth", "io_type"},
        )
        self.assertEqual(encoded.numerical.shape, (6, 2))
        self.assertEqual(encoded.categorical.shape, (6, 2))
        self.assertFalse(encoded.numerical_available[3, 0])
        self.assertFalse(encoded.categorical_available[5, 1])
        self.assertEqual(self.preprocessor.constant_values["constant"], "ssd")

        recovered = self.preprocessor.inverse_transform(
            encoded.numerical, encoded.categorical
        )
        self.assertTrue((recovered["iops"] >= 10).all())
        self.assertTrue((recovered["iops"] <= 60).all())

    def test_log_continuous_transform_round_trip_and_serialization(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "iops"] = 0
        preprocessor = TablePreprocessor(
            ["iops", "lat"],
            ["id"],
            log_continuous_columns=["iops", "lat"],
            log_epsilon=1e-4,
        ).fit(frame)
        encoded = preprocessor.transform(frame)
        recovered = preprocessor.inverse_transform(
            encoded.numerical, encoded.categorical
        )
        for column in ["iops", "lat"]:
            observed = frame[column].notna()
            np.testing.assert_allclose(
                recovered.loc[observed, column].to_numpy(dtype=float),
                frame.loc[observed, column].to_numpy(dtype=float),
                rtol=1e-5,
                atol=1e-5,
            )

        restored = TablePreprocessor.from_dict(preprocessor.to_dict())
        self.assertEqual(restored.log_continuous_columns, ["iops", "lat"])
        self.assertEqual(restored.log_epsilon, 1e-4)
        np.testing.assert_allclose(
            restored.transform(frame).numerical,
            encoded.numerical,
        )

    def test_log_transform_rejects_values_outside_domain(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "iops"] = -1
        with self.assertRaisesRegex(ValueError, "log\\(x \\+ eps\\)"):
            TablePreprocessor(
                ["iops", "lat"],
                ["id"],
                log_continuous_columns=["iops"],
                log_epsilon=1e-6,
            ).fit(frame)

    def test_loss_signs_move_predictions_toward_targets(self) -> None:
        prediction = torch.tensor([2.0, -2.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0])
        supervised = torch.tensor([True, True])
        numerical_loss = masked_mse(prediction, target, supervised)
        numerical_loss.backward()
        self.assertGreater(prediction.grad[0].item(), 0)
        self.assertLess(prediction.grad[1].item(), 0)

        good_logits = torch.tensor([[4.0, -4.0]])
        bad_logits = torch.tensor([[-4.0, 4.0]])
        class_target = torch.tensor([0])
        class_mask = torch.tensor([True])
        self.assertLess(
            masked_cross_entropy(good_logits, class_target, class_mask),
            masked_cross_entropy(bad_logits, class_target, class_mask),
        )

        logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        categorical_loss = masked_cross_entropy(logits, class_target, class_mask)
        categorical_loss.backward()
        self.assertLess(logits.grad[0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 1].item(), 0)

    def test_hybrid_denoiser_round_trip_configuration_and_shapes(self) -> None:
        model = MaskConditionedDenoiser(
            numerical_features=2,
            categorical_sizes=[3, 2],
            d_model=8,
            layers=1,
            heads=1,
            dropout=0.0,
            ff_multiplier=2,
            architecture="hybrid",
            mlp_width=16,
        )
        numerical = torch.randn(4, 2)
        categorical = torch.tensor([[0, 1], [2, 0], [1, 1], [0, 0]])
        time = torch.rand(4)
        target_mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )

        numerical_prediction, categorical_predictions = model(
            numerical, categorical, time, target_mask
        )
        self.assertEqual(numerical_prediction.shape, (4, 2))
        self.assertEqual(
            [prediction.shape for prediction in categorical_predictions],
            [(4, 3), (4, 2)],
        )
        self.assertEqual(model.config()["architecture"], "hybrid")
        self.assertEqual(model.config()["mlp_width"], 16)

        restored = MaskConditionedDenoiser(**model.config())
        restored.load_state_dict(model.state_dict())

    def test_legacy_transformer_checkpoint_configuration_still_loads(self) -> None:
        model = MaskConditionedDenoiser(
            numerical_features=1,
            categorical_sizes=[2],
            d_model=8,
            layers=1,
            heads=1,
            dropout=0.0,
        )
        config = model.config()
        self.assertNotIn("architecture", config)
        restored = MaskConditionedDenoiser(**config)
        restored.load_state_dict(model.state_dict())
        self.assertEqual(restored.architecture, "transformer")

        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            MixedDiffusion(
                MaskConditionedDenoiser(1, [], d_model=8, layers=1, heads=1),
                steps=2,
                categorical_loss_weight=-1.0,
            )

    def test_mlp_denoiser_round_trip_configuration_and_shapes(self) -> None:
        model = MaskConditionedDenoiser(
            numerical_features=2,
            categorical_sizes=[3, 2],
            d_model=7,
            layers=2,
            heads=3,
            dropout=0.0,
            ff_multiplier=2,
            architecture="mlp",
            mlp_width=16,
        )
        numerical = torch.randn(4, 2)
        categorical = torch.tensor([[0, 1], [2, 0], [1, 1], [0, 0]])
        time = torch.rand(4)
        target_mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )

        numerical_prediction, categorical_predictions = model(
            numerical, categorical, time, target_mask
        )
        self.assertEqual(numerical_prediction.shape, (4, 2))
        self.assertEqual(
            [prediction.shape for prediction in categorical_predictions],
            [(4, 3), (4, 2)],
        )
        self.assertFalse(
            any(
                isinstance(module, torch.nn.TransformerEncoder)
                for module in model.modules()
            )
        )
        self.assertEqual(model.config()["architecture"], "mlp")
        self.assertEqual(model.config()["layers"], 2)
        self.assertEqual(model.config()["mlp_width"], 16)

        restored = MaskConditionedDenoiser(**model.config())
        restored.load_state_dict(model.state_dict())

    def test_unseen_category_is_not_used_as_context_or_target(self) -> None:
        train = self.frame.iloc[:4].copy()
        preprocessor = TablePreprocessor(["iops", "lat"], ["id"]).fit(train)
        test = self.frame.iloc[[4]].copy()
        test.loc[test.index[0], "io_type"] = "trim"
        encoded = preprocessor.transform(test)
        io_type_index = preprocessor.categorical_columns.index("io_type")
        self.assertFalse(encoded.categorical_available[0, io_type_index])
        self.assertEqual(
            encoded.categorical[0, io_type_index],
            len(preprocessor.categories["io_type"]),
        )

    def test_artificial_targets_are_observed_and_leave_context(self) -> None:
        available = torch.ones((64, 7), dtype=torch.bool)
        target, mode = sample_artificial_mask(
            available,
            MaskDistribution(),
            generator=torch.Generator().manual_seed(11),
            semantic_groups=[[0, 1], [2, 3, 4]],
        )
        self.assertEqual(mode, "mcar")
        self.assertTrue(torch.all(target <= available))
        self.assertTrue(torch.all(target.sum(dim=1) >= 1))
        self.assertTrue(torch.all(target.sum(dim=1) < available.shape[1]))

    def test_default_mask_distribution_is_mcar_only(self) -> None:
        distribution = MaskDistribution()
        self.assertEqual(
            distribution.probabilities,
            {
                "single": 0.0,
                "mcar": 1.0,
                "loco": 0.0,
                "semantic": 0.0,
                "stress": 0.0,
            },
        )
        self.assertEqual(distribution.min_rate, 0.30)
        self.assertEqual(distribution.max_rate, 0.30)

    def test_fixed_mcar_rate_is_supported(self) -> None:
        available = torch.ones((4096, 9), dtype=torch.bool)
        target, mode = sample_artificial_mask(
            available,
            MaskDistribution(),
            generator=torch.Generator().manual_seed(19),
        )
        self.assertEqual(mode, "mcar")
        self.assertTrue(torch.all(target.sum(dim=1) >= 1))
        self.assertTrue(torch.all(target.sum(dim=1) <= 8))
        self.assertAlmostEqual(float(target.float().mean()), 0.30, delta=0.03)

    def test_loco_masks_one_shared_column(self) -> None:
        available = torch.ones((16, 7), dtype=torch.bool)
        target, mode = sample_artificial_mask(
            available,
            MaskDistribution(),
            generator=torch.Generator().manual_seed(3),
            mode="loco",
        )
        self.assertEqual(mode, "loco")
        selected_columns = torch.nonzero(target.any(dim=0), as_tuple=False).flatten()
        self.assertEqual(selected_columns.numel(), 1)
        self.assertTrue(target[:, selected_columns[0]].all())

    def test_loco_can_be_restricted_to_one_training_column(self) -> None:
        available = torch.ones((16, 7), dtype=torch.bool)
        for seed in range(16):
            target, mode = sample_artificial_mask(
                available,
                MaskDistribution(),
                generator=torch.Generator().manual_seed(seed),
                loco_indices=[3],
                mode="loco",
            )
            self.assertEqual(mode, "loco")
            self.assertTrue(target[:, 3].all())
            self.assertFalse(target[:, [0, 1, 2, 4, 5, 6]].any())

    def test_loco_can_select_every_nonconstant_modeled_column(self) -> None:
        available = torch.ones((16, 7), dtype=torch.bool)
        selected: set[int] = set()
        for seed in range(256):
            target, _ = sample_artificial_mask(
                available,
                MaskDistribution(),
                generator=torch.Generator().manual_seed(seed),
                mode="loco",
            )
            selected.update(
                torch.nonzero(target.any(dim=0), as_tuple=False).flatten().tolist()
            )
        self.assertEqual(selected, set(range(available.shape[1])))

    def test_semantic_mode_is_disabled_by_default(self) -> None:
        available = torch.ones((16, 7), dtype=torch.bool)
        selected_modes = {
            sample_artificial_mask(
                available,
                MaskDistribution(),
                generator=torch.Generator().manual_seed(seed),
                semantic_groups=[[0, 1]],
            )[1]
            for seed in range(256)
        }
        self.assertNotIn("semantic", selected_modes)

    def test_semantic_mask_uses_only_selected_group(self) -> None:
        available = torch.ones((16, 7), dtype=torch.bool)
        target, mode = sample_artificial_mask(
            available,
            MaskDistribution(),
            generator=torch.Generator().manual_seed(5),
            semantic_groups=[[1, 3]],
            mode="semantic",
        )
        self.assertEqual(mode, "semantic")
        self.assertTrue(target[:, [1, 3]].all())
        self.assertFalse(target[:, [0, 2, 4, 5, 6]].any())

    def test_masks_never_hide_real_missing_cells(self) -> None:
        available = torch.ones((8, 6), dtype=torch.bool)
        available[0, 0] = False
        available[1, 2] = False
        for mode in ["single", "mcar", "loco", "semantic", "stress"]:
            target, _ = sample_artificial_mask(
                available,
                MaskDistribution(),
                generator=torch.Generator().manual_seed(17),
                semantic_groups=[[0, 1], [2, 3]],
                mode=mode,
            )
            self.assertTrue(torch.all(target <= available))
            self.assertTrue(torch.all((available & ~target).sum(dim=1) >= 1))

    def test_mixed_loss_and_arbitrary_cell_sampling(self) -> None:
        encoded = self.preprocessor.transform(self.frame)
        numerical = torch.from_numpy(encoded.numerical)
        categorical = torch.from_numpy(encoded.categorical)
        numerical_available = torch.from_numpy(encoded.numerical_available)
        categorical_available = torch.from_numpy(encoded.categorical_available)

        denoiser = MaskConditionedDenoiser(
            numerical_features=2,
            categorical_sizes=self.preprocessor.categorical_sizes,
            d_model=32,
            layers=1,
            heads=4,
            dropout=0.0,
            ff_multiplier=2,
        )
        diffusion = MixedDiffusion(denoiser, steps=4)
        num_targets = torch.zeros_like(numerical_available)
        cat_targets = torch.zeros_like(categorical_available)
        num_targets[0, 0] = True
        num_targets[1, 1] = True
        cat_targets[2, 0] = True
        cat_targets[3, 1] = True
        losses = diffusion.training_loss(
            numerical,
            categorical,
            numerical_available,
            categorical_available,
            num_targets,
            cat_targets,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()

        sample_num, sample_cat = diffusion.sample(
            numerical,
            categorical,
            ~numerical_available | num_targets,
            ~categorical_available | cat_targets,
        )
        observed_num = ~(~numerical_available | num_targets)
        observed_cat = ~(~categorical_available | cat_targets)
        self.assertTrue(torch.equal(sample_num[observed_num], numerical[observed_num]))
        self.assertTrue(
            torch.equal(sample_cat[observed_cat], categorical[observed_cat])
        )
        self.assertTrue(torch.isfinite(sample_num).all())

    def test_sampler_preserves_context_at_every_reverse_step(self) -> None:
        denoiser = RecordingDenoiser()
        diffusion = MixedDiffusion(denoiser, steps=4)
        numerical = torch.tensor([[1.5, 9.0], [2.5, 8.0]])
        categorical = torch.tensor([[0, 1], [2, 0]])
        numerical_target = torch.tensor([[True, False], [True, False]])
        categorical_target = torch.tensor([[True, False], [True, False]])

        sampled_numerical, sampled_categorical = diffusion.sample(
            numerical,
            categorical,
            numerical_target,
            categorical_target,
        )

        self.assertEqual(len(denoiser.inputs), diffusion.steps)
        for numerical_input, categorical_input in denoiser.inputs:
            self.assertTrue(torch.equal(numerical_input[:, 1], numerical[:, 1]))
            self.assertTrue(torch.equal(categorical_input[:, 1], categorical[:, 1]))
        self.assertTrue(torch.equal(sampled_numerical[:, 1], numerical[:, 1]))
        self.assertTrue(torch.equal(sampled_categorical[:, 1], categorical[:, 1]))

    def test_training_forward_process_does_not_noise_context(self) -> None:
        denoiser = RecordingDenoiser()
        diffusion = MixedDiffusion(denoiser, steps=4)
        numerical = torch.tensor([[1.5, 9.0], [2.5, 8.0]])
        categorical = torch.tensor([[0, 1], [2, 0]])
        available = torch.ones((2, 2), dtype=torch.bool)
        numerical_target = torch.tensor([[True, False], [True, False]])
        categorical_target = torch.tensor([[True, False], [True, False]])

        diffusion.training_loss(
            numerical,
            categorical,
            available,
            available,
            numerical_target,
            categorical_target,
        )

        numerical_input, categorical_input = denoiser.inputs[0]
        self.assertTrue(torch.equal(numerical_input[:, 1], numerical[:, 1]))
        self.assertTrue(torch.equal(categorical_input[:, 1], categorical[:, 1]))

    def test_training_rejects_unobserved_artificial_targets(self) -> None:
        denoiser = RecordingDenoiser()
        diffusion = MixedDiffusion(denoiser, steps=4)
        numerical = torch.zeros((2, 2))
        categorical = torch.zeros((2, 2), dtype=torch.long)
        numerical_available = torch.ones((2, 2), dtype=torch.bool)
        numerical_available[0, 0] = False
        categorical_available = torch.ones((2, 2), dtype=torch.bool)
        numerical_target = torch.zeros((2, 2), dtype=torch.bool)
        numerical_target[0, 0] = True
        categorical_target = torch.zeros((2, 2), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "observed training cells"):
            diffusion.training_loss(
                numerical,
                categorical,
                numerical_available,
                categorical_available,
                numerical_target,
                categorical_target,
            )

    def test_context_integrity_report_detects_only_context_changes(self) -> None:
        reference = pd.DataFrame(
            {"iops": [10.0, 20.0], "io_type": ["read", "write"], "id": [1, 2]}
        )
        reconstructed = reference.copy()
        reconstructed.loc[0, "iops"] = 99.0
        reconstructed.loc[1, "io_type"] = "read"
        target_mask = np.array([[True, False], [False, False]], dtype=bool)
        records = context_integrity_records(
            reference,
            reconstructed,
            ["iops", "io_type"],
            target_mask,
            "loco_iops",
            "point_estimate",
        )
        mismatches = {row["column"]: row["mismatches"] for row in records}
        self.assertEqual(mismatches["iops"], 0)
        self.assertEqual(mismatches["io_type"], 1)
        self.assertEqual(mismatches["id"], 0)

    def test_distribution_quality_metrics_are_finite(self) -> None:
        generator = np.random.default_rng(19)

        def make_frame(rows: int) -> pd.DataFrame:
            value = generator.normal(size=rows)
            return pd.DataFrame(
                {
                    "value": value,
                    "related": 2 * value + generator.normal(scale=0.2, size=rows),
                    "category": np.where(value > 0, "positive", "negative"),
                }
            )

        train = make_frame(80)
        real = make_frame(40)
        generated = real.copy()
        generated.loc[::3, "value"] += generator.normal(scale=0.3, size=14)
        metrics, errors = evaluate_generative_quality(
            real,
            generated,
            train,
            sample_size=40,
            dcr_sample_size=20,
            seed=7,
        )
        self.assertFalse(errors)
        self.assertEqual(
            set(metrics),
            {
                "column_shapes",
                "column_pair_trends",
                "overall_quality",
                "alpha_precision",
                "beta_recall",
                "authenticity",
                "c2st",
                "dcr",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_masked_numerical_accuracy_uses_only_target_cells(self) -> None:
        real = pd.DataFrame({"iops": [1.0, 2.0, 3.0], "lat": [10.0, 20.0, 30.0]})
        generated = pd.DataFrame({"iops": [1.0, 4.0, 100.0], "lat": [5.0, 20.0, 25.0]})
        mask = np.array([[False, True], [True, False], [False, True]], dtype=bool)
        metrics = evaluate_masked_numerical_accuracy(
            real,
            generated,
            mask,
            model_columns=["iops", "lat"],
            numerical_columns=["iops", "lat"],
        )
        self.assertEqual(metrics["iops_masked_cells"], 1)
        self.assertEqual(metrics["iops_mse"], 4.0)
        self.assertEqual(metrics["lat_masked_cells"], 2)
        self.assertEqual(metrics["lat_mse"], 25.0)


if __name__ == "__main__":
    unittest.main()
