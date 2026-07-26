from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from ..state import (
    SCHEMA_VERSION,
    ActivationStats,
    CurvatureStats,
    ModuleCalibrationStats,
)

ARTIFACT_KIND = "hif4_ptq_calibration"


def build_calibration_payload(
    modules: dict[str, ModuleCalibrationStats],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, module in modules.items():
        module.validate()
        records[name] = {
            "module_name": name,
            "in_features": int(module.in_features),
            "activation": None if module.activation is None else module.activation.to_record(),
            "curvature": (
                None if module.curvature is None else module.curvature.to_record(store_sum=True)
            ),
        }
    return {
        "__meta__": {
            "kind": ARTIFACT_KIND,
            "schema_version": SCHEMA_VERSION,
            **(metadata or {}),
        },
        "modules": records,
    }


def parse_calibration_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, ModuleCalibrationStats]]:
    meta = dict(payload.get("__meta__", {}))
    if meta.get("kind") != ARTIFACT_KIND:
        raise ValueError(f"unexpected artifact kind={meta.get('kind')!r}")
    if int(meta.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported calibration artifact schema")

    modules: dict[str, ModuleCalibrationStats] = {}
    for name, record in payload.get("modules", {}).items():
        activation_record = record.get("activation")
        curvature_record = record.get("curvature")
        modules[name] = ModuleCalibrationStats(
            module_name=name,
            in_features=int(record["in_features"]),
            activation=(
                None if activation_record is None else ActivationStats.from_record(activation_record)
            ),
            curvature=(
                None if curvature_record is None else CurvatureStats.from_record(curvature_record)
            ),
        )
        modules[name].validate()
    return meta, modules


def save_calibration_artifact(
    path: str | Path,
    modules: dict[str, ModuleCalibrationStats],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_calibration_payload(modules, metadata=metadata), target)


def load_calibration_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, ModuleCalibrationStats]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("calibration artifact must contain a dict payload")
    return parse_calibration_payload(payload)


def merge_calibration_modules(
    shards: Iterable[dict[str, ModuleCalibrationStats]],
) -> dict[str, ModuleCalibrationStats]:
    merged: dict[str, ModuleCalibrationStats] = {}
    for shard in shards:
        for name, incoming in shard.items():
            incoming.validate()
            if name not in merged:
                merged[name] = incoming
                continue
            current = merged[name]
            if current.in_features != incoming.in_features:
                raise ValueError(f"{name}: in_features mismatch across shards")

            activation = current.activation
            if incoming.activation is not None:
                if activation is None:
                    activation = incoming.activation
                else:
                    activation = ActivationStats(
                        absmax=torch.maximum(
                            activation.absmax.float(), incoming.activation.absmax.float()
                        ).cpu(),
                        num_calls=activation.num_calls + incoming.activation.num_calls,
                        channels_dim=activation.channels_dim,
                        module_name=name,
                    )

            curvature = current.curvature
            if incoming.curvature is not None:
                if curvature is None:
                    curvature = incoming.curvature
                else:
                    if curvature.domain != incoming.curvature.domain:
                        raise ValueError(f"{name}: curvature domain mismatch")
                    total_samples = curvature.nsamples + incoming.curvature.nsamples
                    total_sum = curvature.as_sum() + incoming.curvature.as_sum()
                    curvature = CurvatureStats(
                        matrix=total_sum.cpu().contiguous(),
                        nsamples=total_samples,
                        in_features=current.in_features,
                        normalization="sum_xtx",
                        domain=curvature.domain,
                        module_name=name,
                    )

            merged[name] = ModuleCalibrationStats(
                module_name=name,
                in_features=current.in_features,
                activation=activation,
                curvature=curvature,
            )
    return merged


def merge_calibration_artifacts(
    paths: Iterable[str | Path],
) -> tuple[dict[str, Any], dict[str, ModuleCalibrationStats]]:
    all_modules: list[dict[str, ModuleCalibrationStats]] = []
    source_meta: list[dict[str, Any]] = []
    for path in paths:
        meta, modules = load_calibration_artifact(path)
        source_meta.append(meta)
        all_modules.append(modules)
    return {
        "kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "merged_sources": len(source_meta),
        "source_metadata": source_meta,
    }, merge_calibration_modules(all_modules)
