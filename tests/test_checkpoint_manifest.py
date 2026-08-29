import hashlib
import json

import _round3_bootstrap  # noqa: F401

from experiments.round3.artifact_paths import CHECKPOINT_DIR, CHECKPOINT_MANIFEST_DIR


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v3_manifests_match_all_checkpoint_hashes():
    for seed in (7, 17, 27, 37, 47):
        manifest = json.loads(
            (CHECKPOINT_MANIFEST_DIR / f"l1v3_seed{seed}.json").read_text()
        )
        checkpoint = CHECKPOINT_DIR / f"l1v3_seed{seed}_best.pt"
        assert checkpoint.is_file()
        assert len(manifest["checkpoint_sha256"]) == 64
        assert _sha256(checkpoint) == manifest["checkpoint_sha256"]
