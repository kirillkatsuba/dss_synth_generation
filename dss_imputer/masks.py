from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

MaskMode = Literal["single", "mcar", "loco", "semantic", "stress"]


@dataclass(frozen=True)
class MaskDistribution:
    """Artificial missingness distribution; MCAR-only unless overridden."""

    single_probability: float = 0.00
    mcar_probability: float = 1.00
    loco_probability: float = 0.00
    semantic_probability: float = 0.00
    stress_probability: float = 0.00
    min_rate: float = 0.30
    max_rate: float = 0.30
    stress_min_rate: float = 0.50
    stress_max_rate: float = 0.80
    min_context: int = 1

    def __post_init__(self) -> None:
        probabilities = self.probabilities
        if any(value < 0 for value in probabilities.values()):
            raise ValueError("Mask probabilities cannot be negative.")
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("Mask probabilities must sum to one.")
        if not 0 < self.min_rate <= self.max_rate < 1:
            raise ValueError("MCAR rates must satisfy 0 < min <= max < 1.")
        if not 0 < self.stress_min_rate <= self.stress_max_rate < 1:
            raise ValueError("Stress rates must satisfy 0 < min <= max < 1.")
        if self.min_context < 1:
            raise ValueError("min_context must be positive.")

    @property
    def probabilities(self) -> dict[MaskMode, float]:
        return {
            "single": self.single_probability,
            "mcar": self.mcar_probability,
            "loco": self.loco_probability,
            "semantic": self.semantic_probability,
            "stress": self.stress_probability,
        }


def _choose_mode(
    distribution: MaskDistribution,
    generator: torch.Generator | None,
    has_semantic_groups: bool,
) -> MaskMode:
    probabilities = distribution.probabilities.copy()
    if not has_semantic_groups:
        probabilities["mcar"] += probabilities["semantic"]
        probabilities["semantic"] = 0.0

    draw = float(torch.rand((), generator=generator))
    cumulative = 0.0
    for mode, probability in probabilities.items():
        cumulative += probability
        if draw < cumulative:
            return mode
    return "stress"


def _random_rate(
    minimum: float,
    maximum: float,
    generator: torch.Generator | None,
) -> float:
    if minimum == maximum:
        return minimum
    return float(torch.empty(()).uniform_(minimum, maximum, generator=generator))


def _eligible_indices(available_row: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(available_row, as_tuple=False).flatten()


def _random_subset(
    indices: torch.Tensor,
    count: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    order = torch.randperm(indices.numel(), generator=generator)
    return indices[order[:count]]


def _max_targets(available_count: int, min_context: int) -> int:
    return max(0, available_count - min_context)


def _sample_independent_rows(
    available: torch.Tensor,
    minimum_rate: float,
    maximum_rate: float,
    min_context: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    result = torch.zeros_like(available)
    for row in range(available.shape[0]):
        eligible = _eligible_indices(available[row])
        maximum = _max_targets(int(eligible.numel()), min_context)
        if maximum < 1:
            continue
        rate = _random_rate(minimum_rate, maximum_rate, generator)
        selected = torch.rand(eligible.numel(), generator=generator) < rate
        count = max(1, min(int(selected.sum()), maximum))
        result[row, _random_subset(eligible, count, generator)] = True
    return result


def _sample_single(
    available: torch.Tensor,
    min_context: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    result = torch.zeros_like(available)
    for row in range(available.shape[0]):
        eligible = _eligible_indices(available[row])
        if _max_targets(int(eligible.numel()), min_context) < 1:
            continue
        result[row, _random_subset(eligible, 1, generator)] = True
    return result


def _sample_loco(
    available: torch.Tensor,
    min_context: int,
    generator: torch.Generator | None,
    candidate_indices: list[int] | None = None,
) -> torch.Tensor:
    """Hide one common column wherever a row retains enough context."""
    valid_rows = available.sum(dim=1) > min_context
    available_columns = torch.nonzero(
        (available & valid_rows.unsqueeze(1)).any(dim=0), as_tuple=False
    ).flatten()
    if candidate_indices is None:
        candidate_columns = available_columns
    else:
        allowed = torch.zeros(available.shape[1], dtype=torch.bool)
        allowed[candidate_indices] = True
        candidate_columns = available_columns[allowed[available_columns]]
    if candidate_columns.numel() == 0:
        return torch.zeros_like(available)
    column = int(_random_subset(candidate_columns, 1, generator)[0])
    result = torch.zeros_like(available)
    result[:, column] = available[:, column] & valid_rows
    return result


def _sample_semantic_group(
    available: torch.Tensor,
    semantic_groups: list[list[int]],
    min_context: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    group_index = int(torch.randint(0, len(semantic_groups), (), generator=generator))
    group = torch.tensor(semantic_groups[group_index], dtype=torch.long)
    result = torch.zeros_like(available)
    for row in range(available.shape[0]):
        eligible = _eligible_indices(available[row])
        maximum = _max_targets(int(eligible.numel()), min_context)
        if maximum < 1:
            continue
        candidates = group[available[row, group]]
        if candidates.numel() == 0:
            continue
        count = min(int(candidates.numel()), maximum)
        result[row, _random_subset(candidates, count, generator)] = True
    return result


def sample_artificial_mask(
    available: torch.Tensor,
    distribution: MaskDistribution,
    generator: torch.Generator | None = None,
    semantic_groups: list[list[int]] | None = None,
    loco_indices: list[int] | None = None,
    mode: MaskMode | None = None,
) -> tuple[torch.Tensor, MaskMode]:
    """Sample a batch mask while retaining clean context in every usable row.

    A single mode is chosen for the batch. LOCO and semantic modes therefore
    represent a shared conditional task, while single, MCAR, and stress modes
    may choose different target cells for individual rows.
    """
    if available.ndim != 2 or available.dtype != torch.bool:
        raise ValueError("available must be a two-dimensional boolean tensor.")
    if semantic_groups is None:
        semantic_groups = []
    for group in semantic_groups:
        if not group:
            raise ValueError("Semantic mask groups cannot be empty.")
        if min(group) < 0 or max(group) >= available.shape[1]:
            raise ValueError(f"Semantic group contains an invalid index: {group}")
    if loco_indices is not None:
        if not loco_indices:
            raise ValueError("LOCO candidate indices cannot be empty.")
        if min(loco_indices) < 0 or max(loco_indices) >= available.shape[1]:
            raise ValueError(f"LOCO candidates contain an invalid index: {loco_indices}")

    selected_mode = mode or _choose_mode(
        distribution, generator, has_semantic_groups=bool(semantic_groups)
    )
    if selected_mode == "single":
        result = _sample_single(available, distribution.min_context, generator)
    elif selected_mode == "mcar":
        result = _sample_independent_rows(
            available,
            distribution.min_rate,
            distribution.max_rate,
            distribution.min_context,
            generator,
        )
    elif selected_mode == "loco":
        result = _sample_loco(
            available,
            distribution.min_context,
            generator,
            candidate_indices=loco_indices,
        )
    elif selected_mode == "semantic":
        if not semantic_groups:
            raise ValueError("Semantic mode requires at least one semantic group.")
        result = _sample_semantic_group(
            available, semantic_groups, distribution.min_context, generator
        )
    elif selected_mode == "stress":
        result = _sample_independent_rows(
            available,
            distribution.stress_min_rate,
            distribution.stress_max_rate,
            distribution.min_context,
            generator,
        )
    else:
        raise ValueError(f"Unknown mask mode: {selected_mode}")

    return result, selected_mode
