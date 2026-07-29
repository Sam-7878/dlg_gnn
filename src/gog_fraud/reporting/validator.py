from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence_index import verify_evidence_index
from .schema import REPORT_TOP_LEVEL_FIELDS


REQUIRED_MARKDOWN_SECTIONS = tuple([f"## {index}." for index in range(1, 24)] + [f"## Appendix {letter}." for letter in "ABCDEFGH"])


def validate_report(*, report_path: str | Path, json_path: str | Path, evidence_index_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    report_file, json_file = Path(report_path), Path(json_path)
    if not report_file.is_file(): errors.append(f"missing report: {report_file}")
    if not json_file.is_file(): errors.append(f"missing report JSON: {json_file}")
    markdown = report_file.read_text(encoding="utf-8") if report_file.is_file() else ""
    for marker in REQUIRED_MARKDOWN_SECTIONS:
        if marker not in markdown: errors.append(f"missing markdown section: {marker}")
    model: dict[str, Any] = {}
    if json_file.is_file():
        try:
            model = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid report JSON: {exc}")
    for field in REPORT_TOP_LEVEL_FIELDS:
        if field not in model: errors.append(f"missing JSON field: {field}")
    if model:
        md_status = model.get("executive_summary", {}).get("overall_status")
        if md_status and f"`{md_status}`" not in markdown:
            errors.append("Markdown/JSON overall status mismatch")
        if model.get("main_results", {}).get("status") == "NOT_RUN" and "Main Detection Results" in markdown and "`NOT_RUN`" not in markdown:
            errors.append("Markdown hides NOT_RUN main results")
        if model.get("submission_readiness", {}).get("decision") != model.get("executive_summary", {}).get("overall_status"):
            errors.append("readiness decision mismatch within JSON")
        if model.get("consistency_issues"):
            warnings.append(f"{len(model['consistency_issues'])} unresolved consistency issues")
    errors.extend(verify_evidence_index(evidence_index_path, repo_root))
    return {"status": "VALID" if not errors else "INVALID", "errors": errors, "warnings": warnings, "unresolved_warning_count": len(warnings)}
