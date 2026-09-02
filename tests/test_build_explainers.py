import importlib
import json


def test_build_package_mirror_copies_data_and_markdown(tmp_path, monkeypatch):
    # faircode/_explainers/ ships the package's own copy of explainers/*.md +
    # explainers-data.json (issue #388) - a real `pip install faircode[mcp]`
    # never has the repo-root explainers//assets/ directories on disk, so
    # mcp_server.py's list_explainers/get_explainer need this mirror instead.
    script = importlib.import_module("scripts.build_explainers")

    explainers_dir = tmp_path / "explainers"
    explainers_dir.mkdir()
    (explainers_dir / "sample-topic.md").write_text("# Sample Topic\n", encoding="utf-8")
    data_json = tmp_path / "assets" / "explainers-data.json"
    data_json.parent.mkdir()
    entries = [{"slug": "sample-topic", "title": "Sample Topic"}]
    data_json.write_text(json.dumps(entries), encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_explainers"

    monkeypatch.setattr(script, "EXPLAINERS_DIR", explainers_dir)
    monkeypatch.setattr(script, "DATA_JSON", data_json)
    monkeypatch.setattr(script, "PACKAGE_MIRROR_DIR", mirror_dir)

    script.build_package_mirror(entries)

    assert (mirror_dir / "sample-topic.md").read_text(encoding="utf-8") == "# Sample Topic\n"
    assert json.loads((mirror_dir / "data.json").read_text(encoding="utf-8")) == entries


def test_build_package_mirror_removes_a_stale_slug(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.build_explainers")

    explainers_dir = tmp_path / "explainers"
    explainers_dir.mkdir()
    (explainers_dir / "still-here.md").write_text("# Still Here\n", encoding="utf-8")
    data_json = tmp_path / "assets" / "explainers-data.json"
    data_json.parent.mkdir()
    data_json.write_text("[]", encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_explainers"
    mirror_dir.mkdir(parents=True)
    (mirror_dir / "removed-topic.md").write_text("# Gone\n", encoding="utf-8")

    monkeypatch.setattr(script, "EXPLAINERS_DIR", explainers_dir)
    monkeypatch.setattr(script, "DATA_JSON", data_json)
    monkeypatch.setattr(script, "PACKAGE_MIRROR_DIR", mirror_dir)

    script.build_package_mirror([{"slug": "still-here", "title": "Still Here"}])

    assert not (mirror_dir / "removed-topic.md").exists()
    assert (mirror_dir / "still-here.md").is_file()


def test_parse_table_accepts_two_dash_separator_row():
    # explainers/false-positives-vs-false-negatives.md's real separator row
    # is |--|--|--| (2 dashes/cell) - valid GFM, but this parser used to
    # require 3+ dashes and would silently fall through to garbled
    # plain-text lines instead of a real <table> (#324).
    script = importlib.import_module("scripts.build_explainers")
    lines = [
        "| | False Positive | False Negative |",
        "|--|--|--|",
        "| What happens | flags risk | says low risk |",
    ]

    result = script.parse_table(lines, 0)

    assert result is not None
    headers, body_rows, next_index = result
    assert headers == ["", "False Positive", "False Negative"]
    assert body_rows == [["What happens", "flags risk", "says low risk"]]


def test_parse_table_still_accepts_three_dash_separator_row():
    script = importlib.import_module("scripts.build_explainers")
    lines = [
        "| A | B |",
        "|---|---|",
        "| 1 | 2 |",
    ]

    result = script.parse_table(lines, 0)

    assert result is not None
    headers, _body_rows, _next_index = result
    assert headers == ["A", "B"]
