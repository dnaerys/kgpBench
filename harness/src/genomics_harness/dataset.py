"""The Inspect boundary: instances in, `MemoryDataset` and task keyword arguments out.

`MemoryDataset` over explicit `Sample` objects with our own ids. Two hazards are
closed here rather than remembered:

* ``auto_id=True`` overwrites custom ids unconditionally, *after*
  ``record_to_sample`` has run (`dataset/_util.py:104-117`). `MemoryDataset`
  never applies it, which is why the file-backed sources are not used.
* an unset id becomes the sample's position (`_eval/run.py:174-182`), and
  positional ids shift when an instance is inserted at the front — so
  per-instance history across rounds silently misaligns, and retry sample reuse
  is disabled (`_eval/task/run.py:2725-2735`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from .catalogue import Catalogue
from .instances import Instance, InstanceSet
from .provenance import DatasetSnapshot, JudgeConfig, RunProvenance, SolverConfig

__all__ = ["build_dataset", "build_sample", "task_kwargs"]


def build_sample(instance: Instance) -> Sample:
    """One `Sample` for one instance.

    ``target`` stays empty: `Target` is `Sequence[str]` and cannot hold the
    per-check ground-truth structure, which goes to ``metadata`` instead.
    """
    return Sample(
        input=instance.prompt,
        id=instance.id,
        target="",
        metadata=instance.sample_metadata(),
    )


def build_dataset(instances: InstanceSet) -> MemoryDataset:
    """A `MemoryDataset` over an instance set, ids preserved."""
    return MemoryDataset(
        samples=[build_sample(i) for i in instances.instances],
        name=instances.name,
    )


def task_kwargs(
    instances: InstanceSet,
    *,
    catalogue: Catalogue | None = None,
    judges: Sequence[JudgeConfig] = (),
    solver: SolverConfig | None = None,
    dataset_snapshot: DatasetSnapshot | None = None,
) -> dict[str, Any]:
    """The provenance-carrying keyword arguments for a `Task`.

    Returns ``dataset``, ``version`` and ``metadata``, ready to splat:

        Task(solver=..., scorer=[...], **task_kwargs(instances))

    ``version`` is the instance digest, which is what makes `eval_set` treat a
    rotated instance set as new work.
    """
    provenance = RunProvenance.build(
        instances,
        catalogue=catalogue,
        judges=judges,
        solver=solver,
        dataset_snapshot=dataset_snapshot,
    )
    return {
        "dataset": build_dataset(instances),
        "version": provenance.instances_version,
        "metadata": provenance.to_task_metadata(),
    }
