from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import VerificationIssue


def detect_issues(*, repo_root: str | Path, datasets: dict[str, Any], experiments: list[dict[str, Any]], git: dict[str, Any], configs: list[dict[str, Any]]) -> list[VerificationIssue]:
    root = Path(repo_root).resolve()
    issues: list[VerificationIssue] = []
    def add(severity: str, category: str, message: str, evidence: tuple[str, ...] = ()) -> None:
        issues.append(VerificationIssue(f"ISS-{len(issues)+1:03d}", severity, category, message, evidence))

    if not experiments:
        add("CRITICAL", "experiment", "results_sci에 immutable experiment/result row가 없어 detection·routing·calibration·resource·statistics 주장을 검증할 수 없다.")
    missing_chains = sorted({"ethereum", "bsc", "polygon"} - set(datasets))
    if missing_chains:
        add("HIGH", "dataset", f"dataset manifest가 없는 chain: {', '.join(missing_chains)}")
    for chain, manifest in datasets.items():
        ratio = manifest.get("positive_ratio")
        if ratio is not None and float(ratio) > 0.5:
            add("MEDIUM", "label", f"{chain} fraud positive ratio가 {float(ratio):.3f}로 0.5를 초과한다; label semantics와 sampling provenance 확인이 필요하다.", (str(manifest.get("_source", "")),))
    if git.get("dirty"):
        add("HIGH", "provenance", "보고서 생성 기준 working tree가 dirty 상태다; commit SHA만으로 정확한 코드 상태를 재현할 수 없다.")
    manifest_source = root / "src/gog_fraud/data/io/dataset_manifest.py"
    if manifest_source.is_file():
        source = manifest_source.read_text(encoding="utf-8", errors="replace")
        if "cached = hash_index.get(rel)" in source and "stat().st_mtime" not in source and "st_mtime_ns" not in source:
            add("HIGH", "data_hash", "resumable hash index가 path만으로 cache hit를 결정해 원본 파일 변경 후 stale SHA-256을 재사용할 수 있다.", ("src/gog_fraud/data/io/dataset_manifest.py",))
        if "max_files" in source and "truncated" not in source and "scan_complete" not in source:
            add("HIGH", "dataset", "--max-files로 생성한 partial manifest에 incomplete/truncated 표지가 없어 full manifest로 오인될 수 있다.", ("src/gog_fraud/data/io/dataset_manifest.py",))
    if not (root / "results_sci" / "manifests").exists():
        add("HIGH", "provenance", "표준 results_sci/manifests registry와 run_manifest가 없다.")
    if datasets:
        add("MEDIUM", "data_hash", "manifest 내부 raw-file SHA-256 목록은 존재하지만 이번 검증에서 14,464개 원본 파일 전체를 재해싱하지 않았으므로 externally verified 상태가 아니다.")
    if not list((root / "configs/sci").rglob("*.yaml")) if (root / "configs/sci").exists() else True:
        add("HIGH", "config", "SCI config가 없다.")
    elif len(configs) < 2:
        add("MEDIUM", "config", "SCI config가 main 1개뿐이며 data/model/routing/hardware별 immutable snapshot이 없다.")
    if not (root / "requirements-lock.txt").is_file() and not (root / "poetry.lock").is_file():
        add("MEDIUM", "environment", "재현 가능한 Python dependency lock이 없다.")
    if not list(root.glob("**/*split*manifest*.json")):
        add("HIGH", "temporal", "고정 split manifest/hash artifact가 없어 실제 experiment temporal boundary를 검증할 수 없다.")
    return issues
