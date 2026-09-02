"""Tests for the MCP server's tool-logic functions (needs the mcp extra).

These exercise the plain `_*_impl` functions directly rather than going
through a live stdio session - the module docstring explains why (keeps the
`mcp` SDK's own API out of the unit of testing). server-registration/schema
smoke coverage is one small test at the bottom.

Run from the repo root:  pytest tests/ -q
"""

import importlib.util

import pytest

pytest.importorskip("mcp", reason="MCP tools need the optional mcp extra")

from faircode.mcp_server import (  # noqa: E402
    _compare_datasets_impl,
    _profile_dataset_impl,
    _proxy_hints_impl,
    build_server,
)

requires_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="optional 'proxy' extra not installed",
)


def test_profile_dataset_matches_the_shape_profile_returns(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\n" + "M\n" * 8 + "F\n" * 2, encoding="utf-8")

    result = _profile_dataset_impl(str(path))

    assert set(result) == {"n_rows", "n_cols", "overall_score", "grade",
                            "dimensions", "intersections", "flags", "provenance"}
    assert result["n_rows"] == 10


def test_profile_dataset_provenance_default_on_and_matches_the_file_hash(tmp_path):
    import hashlib

    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path))

    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["provenance"]["dataset_hash"] == expected
    assert result["provenance"]["engine"] == "python"


def test_profile_dataset_include_provenance_false_omits_the_block(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path), include_provenance=False)

    assert "provenance" not in result


def test_profile_dataset_unknown_file_raises_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        _profile_dataset_impl(str(tmp_path / "does-not-exist.csv"))


def test_profile_dataset_unknown_override_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overrides column"):
        _profile_dataset_impl(str(path), overrides={"not_a_real_column": "sex"})


def test_profile_dataset_invalid_override_kind_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid kind"):
        _profile_dataset_impl(str(path), overrides={"sex": "not_a_real_kind"})


def test_profile_dataset_cross_with_one_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex,race\nM,W\nF,B\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cross expects exactly two"):
        _profile_dataset_impl(str(path), cross=["sex"])


def test_profile_dataset_cross_unknown_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex,race\n" + "M,W\nF,B\n" * 25, encoding="utf-8")

    with pytest.raises(ValueError, match="raace"):
        _profile_dataset_impl(str(path), cross=["sex", "raace"])


def test_profile_dataset_reference_path_adds_reference_hash_to_provenance(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\n" + "M\n" * 5 + "F\n" * 5, encoding="utf-8")
    ref = tmp_path / "ref.csv"
    ref.write_text("column,group,share\nsex,M,0.5\nsex,F,0.5\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path), reference_path=str(ref))

    assert "reference_hash" in result["provenance"]
    assert result["dimensions"][0]["reference"] is not None


def test_compare_datasets_matches_the_shape_compare_returns(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\n" + "M\n" * 8 + "F\n" * 2, encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\n" + "M\n" * 5 + "F\n" * 5, encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b))

    assert set(result) >= {"a", "b", "score_delta", "dimensions",
                           "added_dimensions", "removed_dimensions", "flags"}
    assert "provenance" in result


def test_compare_datasets_provenance_has_both_dataset_hashes(tmp_path):
    import hashlib

    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\nM\nF\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\nM\nM\n", encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b))

    assert result["provenance"]["dataset_hash_a"] == "sha256:" + hashlib.sha256(path_a.read_bytes()).hexdigest()
    assert result["provenance"]["dataset_hash_b"] == "sha256:" + hashlib.sha256(path_b.read_bytes()).hexdigest()


def test_compare_datasets_unknown_override_column_raises(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\nM\nF\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\nM\nM\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overrides column"):
        _compare_datasets_impl(str(path_a), str(path_b), overrides={"nope": "sex"})


def test_compare_datasets_proxy_hints_defaults_off(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\nM\nF\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\nM\nM\n", encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b))

    assert "proxy_hints_a" not in result
    assert "proxy_hints_b" not in result


@requires_scipy
def test_compare_datasets_proxy_hints_attaches_hints_for_both_datasets(tmp_path):
    # occupation is a perfect function of sex in both files -> maximal association.
    rows = ["sex,occupation"] + [
        f"{'male' if i % 2 == 0 else 'female'},{'engineer' if i % 2 == 0 else 'nurse'}"
        for i in range(100)
    ]
    path_a = tmp_path / "a.csv"
    path_a.write_text("\n".join(rows) + "\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b), proxy_hints=True)

    for key in ("proxy_hints_a", "proxy_hints_b"):
        pair = next(h for h in result[key] if {h["a"], h["b"]} == {"sex", "occupation"})
        assert pair["p_value"] < 0.05


@requires_scipy
def test_proxy_hints_returns_a_dict_with_a_hints_key(tmp_path):
    # occupation is a perfect function of sex -> maximal association.
    rows = ["sex,occupation"] + [
        f"{'male' if i % 2 == 0 else 'female'},{'engineer' if i % 2 == 0 else 'nurse'}"
        for i in range(100)
    ]
    path = tmp_path / "a.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = _proxy_hints_impl(str(path))

    assert list(result) == ["hints"]
    pair = next(h for h in result["hints"] if {h["a"], h["b"]} == {"sex", "occupation"})
    assert pair["p_value"] < 0.05


def test_proxy_hints_with_no_significant_pairs_returns_an_empty_list_not_an_error(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _proxy_hints_impl(str(path))

    assert result == {"hints": []}


def test_proxy_hints_runtime_error_propagates_with_a_clean_message(tmp_path, monkeypatch):
    import faircode.mcp_server as mcp_server

    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    def raise_runtime(_df, _dimensions, **_kwargs):
        raise RuntimeError("proxy hints need scipy (install with: pip install faircode[proxy])")

    monkeypatch.setattr(mcp_server, "compute_proxy_hints", raise_runtime)

    with pytest.raises(RuntimeError, match="proxy hints need scipy"):
        _proxy_hints_impl(str(path))


@requires_scipy
def test_proxy_hints_held_out_with_flags_a_dropped_column(tmp_path):
    # zip_code is perfectly aligned with race, which has been dropped from
    # the profiled dataset - only visible via held_out_with.
    path = tmp_path / "dropped.csv"
    held_path = tmp_path / "full.csv"
    zip_code = (["111"] * 100 + ["222"] * 100)
    race = (["A"] * 100 + ["B"] * 100)
    path.write_text("zip_code\n" + "\n".join(zip_code), encoding="utf-8")
    held_path.write_text("zip_code,race\n" +
                         "\n".join(f"{z},{r}" for z, r in zip(zip_code, race)),
                         encoding="utf-8")

    result = _proxy_hints_impl(str(path), held_out_with=[f"{held_path}=race"])

    pair = next(h for h in result["hints"] if {h["a"], h["b"]} == {"zip_code", "race"})
    assert pair["p_value"] < 0.05


def test_proxy_hints_held_out_with_malformed_spec_raises_value_error(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid held_out_with 'noequalssign'"):
        _proxy_hints_impl(str(path), held_out_with=["noequalssign"])


def test_proxy_hints_held_out_with_column_collision_raises_value_error(tmp_path):
    path = tmp_path / "a.csv"
    held_path = tmp_path / "b.csv"
    path.write_text("sex,race\nM,A\nF,B\n", encoding="utf-8")
    held_path.write_text("race\nX\nY\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column 'race' already exists"):
        _proxy_hints_impl(str(path), held_out_with=[f"{held_path}=race"])


def test_build_server_registers_all_three_phase_one_tools():
    import asyncio

    server = build_server()
    tools = asyncio.run(server.list_tools())

    assert {t.name for t in tools} == {"profile_dataset", "compare_datasets", "proxy_hints"}


def test_tool_errors_surface_the_anticipated_message_not_a_generic_one():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool("profile_dataset", {"path": "definitely-missing.csv"})

    with pytest.raises(ToolError, match="file not found: definitely-missing.csv"):
        asyncio.run(call())


def test_compare_datasets_tool_error_surfaces_via_call_tool():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool(
            "compare_datasets", {"path_a": "definitely-missing.csv", "path_b": "also-missing.csv"})

    with pytest.raises(ToolError, match="file not found: definitely-missing.csv"):
        asyncio.run(call())


def test_proxy_hints_tool_error_surfaces_via_call_tool():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool("proxy_hints", {"path": "definitely-missing.csv"})

    with pytest.raises(ToolError, match="file not found: definitely-missing.csv"):
        asyncio.run(call())
