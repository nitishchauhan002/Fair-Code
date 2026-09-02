"""Tests for scripts/freeze_paper_results.py's mirror_for_mcp() - the small,
always-safe piece of this script (copies FROZEN_DIR -> MCP_MIRROR_DIR, never
touches results/ or paper/results-frozen/ itself). The full freeze() flow
needs a real results/ directory and git state, out of scope here.
"""

import importlib


def test_mirror_for_mcp_copies_the_three_frozen_csvs(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.freeze_paper_results")

    frozen_dir = tmp_path / "paper" / "results-frozen"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "results_fairness.csv").write_text("audit,metric\ncompas,dp\n", encoding="utf-8")
    (frozen_dir / "results_performance.csv").write_text("audit,metric\ncompas,acc\n", encoding="utf-8")
    (frozen_dir / "summary.csv").write_text("audit\ncompas\n", encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_results_frozen"

    monkeypatch.setattr(script, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(script, "MCP_MIRROR_DIR", mirror_dir)

    script.mirror_for_mcp()

    for name in ("results_fairness.csv", "results_performance.csv", "summary.csv"):
        assert (mirror_dir / name).read_text(encoding="utf-8") == (frozen_dir / name).read_text(encoding="utf-8")


def test_mirror_for_mcp_never_writes_to_frozen_dir(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.freeze_paper_results")

    frozen_dir = tmp_path / "paper" / "results-frozen"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "results_fairness.csv").write_text("audit\ncompas\n", encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_results_frozen"

    monkeypatch.setattr(script, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(script, "MCP_MIRROR_DIR", mirror_dir)

    before = {p.name: p.read_text(encoding="utf-8") for p in frozen_dir.iterdir()}
    script.mirror_for_mcp()
    after = {p.name: p.read_text(encoding="utf-8") for p in frozen_dir.iterdir()}

    assert before == after
