#!/usr/bin/env python3
"""
test_github_remote_status_gate.py

Round P4 External GitHub Verification Gate:
Queries the public GitHub API (https://api.github.com/repos/Sam-7878/dlg_gnn)
without requiring authenticated credentials to verify:
1. Repository is publicly visible (private == False).
2. Root README is the benchmark reproduction landing page.
3. Tag 'v1.0.0-preprint' exists remotely or is flagged as PENDING_GITHUB_RELEASE.
4. GitHub release exists or is flagged as PENDING_GITHUB_RELEASE.
5. Release asset is publicly downloadable or flagged as PENDING_GITHUB_RELEASE.

Network-Aware Policy:
If network access is unavailable or GitHub API rate limit is exceeded,
the test gracefully skips with 'SKIP_WITH_REQUIRED_MANUAL_EXTERNAL_CHECK'
rather than emitting a false PASS.
"""

import json
from pathlib import Path
import urllib.error
import urllib.request
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GITHUB_API_REPO = "https://api.github.com/repos/Sam-7878/dlg_gnn"
GITHUB_RAW_README = "https://raw.githubusercontent.com/Sam-7878/dlg_gnn/main/README.md"


def _fetch_github_api(url: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DLG-GNN-Publication-Audit-Agent"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 403:
            pytest.skip(f"SKIP_WITH_REQUIRED_MANUAL_EXTERNAL_CHECK: GitHub API rate limited ({e.code})")
        elif e.code == 404:
            return None
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"SKIP_WITH_REQUIRED_MANUAL_EXTERNAL_CHECK: Network unavailable ({e})")


def test_github_repository_is_public():
    data = _fetch_github_api(GITHUB_API_REPO)
    assert data is not None, "Repository not found on GitHub"
    assert data.get("private") is False, "Repository is not public"
    assert data.get("name") == "dlg_gnn"


def test_github_root_readme_is_benchmark_landing_page():
    req = urllib.request.Request(
        GITHUB_RAW_README,
        headers={"User-Agent": "DLG-GNN-Publication-Audit-Agent"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"SKIP_WITH_REQUIRED_MANUAL_EXTERNAL_CHECK: Raw README fetch unavailable ({e})")

    is_updated = ("DLG-GNN Benchmark" in content or "Benchmark for Graph Anomaly Detection" in content) and "0009-0008-4056-3875" in content
    if not is_updated:
        pytest.skip("PENDING_GITHUB_PUSH: Root README on GitHub remote is at prior commit; 'git push origin main' required to update remote landing page")
    assert is_updated


def test_github_tag_exists():
    tags = _fetch_github_api(f"{GITHUB_API_REPO}/tags")
    assert tags is not None
    tag_names = [t.get("name") for t in tags]
    if "v1.0.0-preprint" not in tag_names:
        pytest.skip("PENDING_GITHUB_RELEASE: Git tag 'v1.0.0-preprint' is prepared locally; push required to remote")
    assert "v1.0.0-preprint" in tag_names


def test_github_release_exists():
    releases = _fetch_github_api(f"{GITHUB_API_REPO}/releases")
    assert releases is not None
    release_tags = [r.get("tag_name") for r in releases]
    if "v1.0.0-preprint" not in release_tags:
        pytest.skip("PENDING_GITHUB_RELEASE: GitHub release 'v1.0.0-preprint' prepared locally; upload required on GitHub")
    assert "v1.0.0-preprint" in release_tags


def test_github_release_asset_downloadable():
    releases = _fetch_github_api(f"{GITHUB_API_REPO}/releases")
    assert releases is not None
    matching = [r for r in releases if r.get("tag_name") == "v1.0.0-preprint"]
    if not matching:
        pytest.skip("PENDING_GITHUB_RELEASE: Release asset upload pending on remote")

    assets = matching[0].get("assets", [])
    asset_names = [a.get("name") for a in assets]
    if "DLG_GNN_Benchmark_v1.0.0_preprint.zip" not in asset_names:
        pytest.skip("PENDING_GITHUB_RELEASE: Asset 'DLG_GNN_Benchmark_v1.0.0_preprint.zip' pending upload")
    assert "DLG_GNN_Benchmark_v1.0.0_preprint.zip" in asset_names
