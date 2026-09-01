"""Download the three benchmarks Memory-R1 is evaluated on.

LoCoMo         — Maharana et al. 2024, ``snap-stanford/locomo`` on HF Hub. The raw file is
                 ``locomo10.json`` (300+ turns per dialogue, single/multi-hop/open-domain/temporal
                 QA). Memory-R1 uses this as its *only training source* (152/81/1307 split), and
                 evaluates zero-shot on MSC and LongMemEval.
MSC            — Multi-Session Chat, Xu et al. 2021, ``facebook/msc``. Memory-R1 uses the
                 memory-eval subset following MemGPT.
LongMemEval    — Wu et al. 2024, ``xiaowu0162/longmemeval`` on HF Hub. Broad long-term memory
                 benchmark; SSU/SSP/OD/MS/KU/TR question types.

We download via ``huggingface_hub.snapshot_download`` (works with anon + private caching). If the
LoCoMo HF Hub name changes, we fall back to the GitHub release URL. Users can pass their own paths
if these move.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()


@dataclass
class DatasetSpec:
    """One benchmark: HF repo (dataset) + fallback URL + local marker file."""

    name: str
    hf_repo: str | None
    hf_type: str  # "dataset" or "space"
    fallback_url: str | None
    marker_file: str  # a file whose presence signals "already downloaded"

    def dest(self, out_dir: Path) -> Path:
        return out_dir / self.name

    def already_downloaded(self, out_dir: Path) -> bool:
        return (self.dest(out_dir) / self.marker_file).exists()


DATASETS: dict[str, DatasetSpec] = {
    "locomo": DatasetSpec(
        # Repo now lives under snap-research (moved from snap-stanford in 2024).
        name="locomo",
        hf_repo="snap-research/locomo",
        hf_type="dataset",
        fallback_url="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        marker_file="locomo10.json",
    ),
    "msc": DatasetSpec(
        # Original MSC (Xu et al., 2021 "Beyond Goldfish Memory") is on ParlAI; this HF mirror
        # ports it to parquet (23.4k rows across 4 sessions per dialogue).
        name="msc",
        hf_repo="nayohan/multi_session_chat",
        hf_type="dataset",
        fallback_url=None,
        marker_file="msc.jsonl",  # we consolidate parquet -> jsonl in _download_msc
    ),
    "longmemeval": DatasetSpec(
        # Official cleaned release: xiaowu0162/longmemeval-cleaned.
        name="longmemeval",
        hf_repo="xiaowu0162/longmemeval-cleaned",
        hf_type="dataset",
        fallback_url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        marker_file="longmemeval_s_cleaned.json",
    ),
}


# --------------------------------------------------------------------------- helpers


def _download_url(url: str, out_path: Path) -> None:
    """Streamed download with a progress bar."""

    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with Progress() as progress:
            task = progress.add_task(f"[cyan]{out_path.name}", total=total or None)
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if not chunk:
                        continue
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))


def _snapshot_from_hf(spec: DatasetSpec, dest: Path) -> bool:
    """Try to fetch via ``huggingface_hub.snapshot_download``. Returns True on success."""

    from huggingface_hub import snapshot_download

    if spec.hf_repo is None:
        return False
    try:
        snapshot_download(
            repo_id=spec.hf_repo,
            repo_type=spec.hf_type,
            local_dir=str(dest),
        )
        return True
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]HF snapshot failed for {spec.name}: {e}[/yellow]")
        return False


# --------------------------------------------------------------------------- LoCoMo


def _download_locomo(dest: Path, spec: DatasetSpec) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if _snapshot_from_hf(spec, dest):
        # snapshot may put the file under a data/ subdir — normalize.
        for candidate in dest.rglob("locomo10.json"):
            if candidate != dest / "locomo10.json":
                shutil.copy(candidate, dest / "locomo10.json")
            break
    if (dest / spec.marker_file).exists():
        return
    if spec.fallback_url:
        console.print(f"[cyan]Falling back to raw URL for LoCoMo: {spec.fallback_url}[/cyan]")
        _download_url(spec.fallback_url, dest / spec.marker_file)
    if not (dest / spec.marker_file).exists():
        raise RuntimeError(
            f"Could not obtain LoCoMo; try downloading manually from "
            f"https://github.com/snap-stanford/locomo and placing locomo10.json in {dest}."
        )


# --------------------------------------------------------------------------- MSC


def _download_msc(dest: Path, spec: DatasetSpec) -> None:
    """Fetch the HF-mirrored MSC parquet + consolidate to ``msc.jsonl``.

    The MSC dataset (Xu et al., 2021 "Beyond Goldfish Memory") is originally distributed via ParlAI
    but requires registration. ``nayohan/multi_session_chat`` mirrors it as parquet on HF Hub with
    the same dialogue_id / session_id / persona1 / persona2 / dialogue / speaker fields, which is
    what MemGPT and Memory-R1 need for evaluation.
    """

    dest.mkdir(parents=True, exist_ok=True)
    if not _snapshot_from_hf(spec, dest):
        console.print(
            f"[yellow]MSC HF snapshot failed. Manually obtain the dataset from "
            f"https://parl.ai/projects/msc/ or "
            f"https://huggingface.co/datasets/{spec.hf_repo} and place the files at {dest}.[/yellow]"
        )
        return

    _consolidate_msc(dest)


def _consolidate_msc(dest: Path) -> None:
    """Merge all parquet files under ``dest`` into a single ``msc.jsonl``."""

    out = dest / "msc.jsonl"
    if out.exists():
        return

    import pandas as pd

    parquets = list(dest.rglob("*.parquet"))
    if not parquets:
        return

    with out.open("w", encoding="utf-8") as f:
        for pq in parquets:
            split = pq.parent.name  # train / validation / test if organized that way
            df = pd.read_parquet(pq)
            for row in df.to_dict(orient="records"):
                row["_split"] = split
                # Normalize sequence fields (numpy arrays -> lists).
                for k, v in list(row.items()):
                    if hasattr(v, "tolist"):
                        row[k] = v.tolist()
                import json as _json

                f.write(_json.dumps(row, ensure_ascii=False, default=str) + "\n")
    console.print(f"[green]Wrote consolidated MSC jsonl → {out}[/green]")


# --------------------------------------------------------------------------- LongMemEval


def _download_longmemeval(dest: Path, spec: DatasetSpec) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if _snapshot_from_hf(spec, dest):
        # Normalize any nested layouts.
        for candidate in dest.rglob(spec.marker_file):
            if candidate != dest / spec.marker_file:
                shutil.copy(candidate, dest / spec.marker_file)
            break
    if not (dest / spec.marker_file).exists():
        console.print(
            "[yellow]LongMemEval marker not found after HF snapshot. Files that were fetched:\n"
            + "\n".join(f"  - {p.relative_to(dest)}" for p in dest.rglob('*') if p.is_file())[:8000]
            + "\nSee https://github.com/xiaowu0162/LongMemEval for manual download.[/yellow]"
        )


# --------------------------------------------------------------------------- entry point


def download_all(dataset: str = "all", out_dir: Path = Path("data/raw")) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    which = list(DATASETS) if dataset == "all" else [dataset]
    for key in which:
        spec = DATASETS[key]
        dest = spec.dest(out_dir)
        if spec.already_downloaded(out_dir):
            console.print(f"[green]{key} already present at {dest}. Skipping.[/green]")
            continue
        console.rule(f"Downloading {key}")
        if key == "locomo":
            _download_locomo(dest, spec)
        elif key == "msc":
            _download_msc(dest, spec)
        elif key == "longmemeval":
            _download_longmemeval(dest, spec)
        else:  # unknown
            raise ValueError(f"Unknown dataset {key}")
        console.print(f"[green]{key} → {dest}[/green]")
