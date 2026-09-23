import pytest

from gog_fraud.reporting.markdown_report import generate_markdown_report
from gog_fraud.reporting.latex_exporter import generate_latex_report


def test_legacy_hard_coded_claim_report_is_disabled(tmp_path):
    with pytest.raises(RuntimeError, match="unverified fixed scientific claims"):
        generate_markdown_report(tmp_path, {}, None, None, None, None, None, None)


def test_legacy_hard_coded_latex_report_is_disabled(tmp_path):
    with pytest.raises(RuntimeError, match="unverified fixed scientific claims"):
        generate_latex_report(tmp_path, None, None, None, None, None, None)
