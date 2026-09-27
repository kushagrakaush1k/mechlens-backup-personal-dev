#!/usr/bin/env python3
"""Measure the checkout's real ActivationCache.__repr__, without printing tensors.

Run from an environment containing this checkout's dependencies:
    python debugging/activation_cache_repr_probe.py
    python debugging/activation_cache_repr_probe.py --repo-root /path/to/TransformerLens

Allocations, imports, warmups and output are outside timing. Each row reports
repeated batch timings divided by calls per batch, not notebook rendering time.
CPU float32 tensors have distinct, initialized storage; only one case is live
at a time. No model download, source rewriting, or fallback implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import timeit
import types
from datetime import datetime, timezone
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/activation-cache-repr-measurements.md"),
        help="Markdown results path, relative to repo root; replaced after a successful run",
    )
    parser.add_argument("--fixed-keys", type=positive_int, default=64)
    parser.add_argument("--tensor-sizes", type=positive_int, nargs="+", default=[1, 1024, 65536, 262144])
    parser.add_argument("--key-counts", type=positive_int, nargs="+", default=[16, 64, 256, 1024, 4096])
    parser.add_argument("--fixed-numel", type=positive_int, default=16)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repeat", type=positive_int, default=9)
    parser.add_argument("--number", type=positive_int, default=100)
    return parser.parse_args()


def load_checkout(repo_root: Path):
    root = repo_root.resolve(strict=True)
    expected = root / "transformer_lens" / "ActivationCache.py"
    if not expected.is_file() or not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            f"Not a complete checkout: expected {expected} and {root / 'pyproject.toml'}. "
            "Put this script in the repository's debugging/ directory or pass --repo-root."
        )
    expected = expected.resolve(strict=True)
    source_bytes = expected.read_bytes()
    # Prefer the checkout, then verify what Python actually imported. Do not
    # evict cached modules: a wrong preloaded module must fail verification.
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    module = importlib.import_module("transformer_lens.ActivationCache")
    from transformer_lens.ActivationCache import ActivationCache

    locations = {
        "module": module.__file__,
        "module spec": module.__spec__.origin,
        "class": inspect.getsourcefile(ActivationCache),
        "repr": inspect.getsourcefile(ActivationCache.__repr__),
    }
    for label, location in locations.items():
        if location is None or Path(location).resolve() != expected:
            raise RuntimeError(f"Wrong {label} source: {location!r}; expected {expected}")
    if ActivationCache.__module__ != module.__name__:
        raise RuntimeError("ActivationCache was re-exported from a different module")

    # Compile only for verification; never execute this code or substitute it
    # for the imported class. Code equality catches a replaced __repr__ even
    # when its reported source filename still points into the checkout.
    loaded_repr = ActivationCache.__dict__.get("__repr__")
    if not inspect.isfunction(loaded_repr):
        raise RuntimeError("Expected a directly defined Python __repr__ method")
    compiled = compile(source_bytes, loaded_repr.__code__.co_filename, "exec")

    def child_code(parent: types.CodeType, name: str) -> types.CodeType:
        matches = [c for c in parent.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely locate {name} in checkout source")
        return matches[0]

    disk_repr = child_code(child_code(compiled, "ActivationCache"), "__repr__")
    if loaded_repr.__code__ != disk_repr:
        raise RuntimeError("Loaded __repr__ code differs from the checkout source")
    if expected.read_bytes() != source_bytes:
        raise RuntimeError("Source changed during import; rerun with a stable checkout")

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Verified source: {expected}")
    print(f"Source SHA-256: {hashlib.sha256(source_bytes).hexdigest()}")
    print("Verified: imported __repr__ code matches the file on disk")
    print(inspect.getsource(loaded_repr).strip())
    return ActivationCache, expected, source_bytes, loaded_repr


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output = (root / args.output).resolve()
    if output.suffix != ".md":
        raise ValueError("--output must be a Markdown (.md) file")
    if output == root / "docs/activation-cache-repr-triage.md":
        raise ValueError("Write a separate measurements file, then review it before updating triage")
    Cache, source_path, source_bytes, original_repr = load_checkout(root)
    import torch

    print(f"PyTorch: {torch.__version__}; CPU float32; distinct storage per key")
    print(f"Warmup: {args.warmup}; batches: {args.repeat}; repr calls/batch: {args.number}")
    print("Times are microseconds per call; all batch samples are shown.")

    rows: list[str] = []
    raw_samples: list[str] = []

    def measure(sweep: str, keys: tuple[str, ...], numel: int) -> None:
        # Real, independently allocated tensors, not expand() views or shared
        # references. Allocation/initialization is deliberately not timed.
        values = {key: torch.ones(numel, dtype=torch.float32, device="cpu") for key in keys}
        cache = Cache(values, model=None)
        payload_mib = sum(t.numel() * t.element_size() for t in values.values()) / 2**20
        for _ in range(args.warmup):
            repr(cache)
        timer = timeit.Timer(lambda: repr(cache))
        samples = [seconds * 1e6 / args.number for seconds in timer.repeat(args.repeat, args.number)]
        chars = len(repr(cache))
        print(
            f"{len(keys):7d} {numel:10d} {payload_mib:11.3f} {chars:10d} "
            f"{min(samples):10.3f} {statistics.median(samples):10.3f} {max(samples):10.3f} "
            + ",".join(f"{sample:.3f}" for sample in samples)
        )
        rows.append(
            f"| {sweep} | {len(keys):,} | {numel:,} | {payload_mib:.6f} | {chars:,} | "
            f"{min(samples):.3f} | {statistics.median(samples):.3f} | {max(samples):.3f} |"
        )
        raw_samples.append(
            f"{sweep}, keys={len(keys)}, numel={numel}: "
            + ", ".join(f"{sample:.6f}" for sample in samples)
        )
        # Locals, including timer's cache closure, are released on return.

    def keys_for(count: int) -> tuple[str, ...]:
        # Fixed-width keys keep per-key string length constant across sweeps.
        return tuple(f"blocks.{i:08d}.hook_resid_pre" for i in range(count))

    header = "   keys      numel payload_MiB repr_chars     min_us  median_us     max_us samples_us"
    fixed_keys = keys_for(args.fixed_keys)
    print("\nA: identical keys; increasing elements per tensor")
    print(header)
    for numel in sorted(set(args.tensor_sizes)):
        measure("A: tensor size", fixed_keys, numel)
    print("\nB: increasing key count; fixed elements per tensor")
    print(header)
    for count in sorted(set(args.key_counts)):
        measure("B: key count", keys_for(count), args.fixed_numel)

    if source_path.read_bytes() != source_bytes or Cache.__repr__ is not original_repr:
        raise RuntimeError("Source or __repr__ changed during measurement; discard these results")
    print("\nCompleted; source and imported __repr__ remained unchanged.")

    def git_info(*git_args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *git_args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "UNAVAILABLE"

    report = [
        "# ActivationCache repr measurements",
        "",
        f"Completed (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Commit: {git_info('rev-parse', 'HEAD')}",
        f"Python: {sys.version.split()[0]} ({sys.executable})",
        f"PyTorch: {torch.__version__}",
        f"OS: {platform.platform()}; CPU: {platform.processor() or platform.machine()}",
        f"Verified source: {source_path}",
        f"Source SHA-256: {hashlib.sha256(source_bytes).hexdigest()}",
        "",
        "Imported repr matched the checkout source; source and method stayed unchanged.",
        f"CPU float32, distinct storage; warmup={args.warmup}, "
        f"batches={args.repeat}, calls/batch={args.number}.",
        "Timings are microseconds per call, summarized across batch averages. "
        "Imports, allocation, warmups, printing, and file output are excluded. "
        "This does not measure notebook rendering or GPU behavior.",
        "Pytest and acceptance-test results: PENDING (not run by this probe).",
        "",
        "| Sweep | Keys | Elements per tensor | Payload MiB | Repr characters | "
        "Min µs/call | Median µs/call | Max µs/call |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Raw batch averages (µs/call)",
        "",
        "```text",
        *raw_samples,
        "```",
        "",
        "## Working-tree status before saving results",
        "",
        "```text",
        git_info("status", "--short") or "(clean)",
        "```",
        "",
        "## Verified repr source",
        "",
        "```python",
        inspect.getsource(original_repr).strip(),
        "```",
        "",
    ]
    # Replace atomically so a failed write cannot leave a partial results file.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write("\n".join(report))
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved measurements: {output}")



if __name__ == "__main__":
    main()
