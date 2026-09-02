"""MCP server exposing the profiler's Python API as tools for an LLM agent to
call directly, instead of shelling out to the CLI and parsing text.

    faircode-mcp                # run directly (stdio transport)
    python -m faircode.mcp_server

Stdio-only - no network listener, no auth, no hosting. Every tool reads a
local file path the calling process already has access to, the same trust
boundary the CLI already has: nothing here is a new capability, just a
different way to call the same profile()/compare()/proxy_hints() functions
cli.py already wraps. See faircode/SPEC.md section 11 for the tool contract.

Needs the optional 'mcp' extra (`pip install faircode[mcp]`).

Tool logic lives in plain, directly-callable `_*_impl` functions (this
module's actual unit of testing - tests/test_mcp_server.py calls these, not
the MCP-decorated wrappers) so `mcp` SDK API churn - it renamed FastMCP to
MCPServer between v1 and v2 - stays contained to `build_server()` and doesn't
leak into anything else. The `_impl` functions raise plain ValueError /
FileNotFoundError / RuntimeError; `build_server()`'s wrappers translate those
into `ToolError` so the anticipated-failure message (e.g. "file not found:
X") actually reaches the calling agent - any other exception type is treated
by the SDK as a crash and replaced with a generic "Error executing tool X",
withholding the real text.
"""

from __future__ import annotations

from . import __version__
from .compare import compare
from .detect import VALID_KINDS
from .loaders_extra import read_table
from .profiler import _resolve_opts, parse_reference, profile
from .provenance import build as build_provenance
from .proxy import parse_held_out_specs
from .proxy import proxy_hints as compute_proxy_hints

_MAP_CHOICES = VALID_KINDS + ("ignore",)


def _read_table_or_raise(path: str):
    """Read a table, translating loader failures into a clear message instead
    of a raw pandas/parser traceback. Mirrors cli.py's _read_or_exit, minus
    the SystemExit - a tool function should raise, not exit the process."""
    try:
        return read_table(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found: {path}") from None
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any parse failure plainly
        raise RuntimeError(f"could not read dataset {path}: {exc}") from exc


def _check_overrides(overrides, known_columns):
    """Mirrors cli.py's _parse_map + _check_map_columns: reject an unknown
    column or an invalid kind instead of detect_columns() silently no-opping
    on it."""
    if not overrides:
        return
    unknown = [col for col in overrides if col not in known_columns]
    if unknown:
        raise ValueError(f"overrides column(s) not found in the dataset: {', '.join(unknown)}")
    bad = {col: kind for col, kind in overrides.items() if kind not in _MAP_CHOICES}
    if bad:
        pairs = ", ".join(f"{col}={kind}" for col, kind in bad.items())
        raise ValueError(f"invalid kind for {pairs}; choose from {', '.join(_MAP_CHOICES)}")


def _build_opts(min_share=None, intersection_floor=None, imbalance_flag=None,
                missing_flag=None, min_group_size=None, cross=None,
                reference_path=None):
    opts = {
        "min_share": min_share,
        "intersection_floor": intersection_floor,
        "imbalance_flag": imbalance_flag,
        "missing_flag": missing_flag,
        "min_group_size": min_group_size,
    }
    if cross:
        if len(cross) != 2 or not all(cross):
            raise ValueError("cross expects exactly two column names")
        opts["cross"] = list(cross)
    if reference_path:
        opts["reference"] = parse_reference(_read_table_or_raise(reference_path))
    return opts


def _profile_dataset_impl(path, overrides=None, cross=None, reference_path=None,
                          min_share=None, intersection_floor=None,
                          imbalance_flag=None, missing_flag=None,
                          min_group_size=None, include_provenance=True):
    overrides = overrides or {}
    df = _read_table_or_raise(path)
    _check_overrides(overrides, df.columns)
    opts = _build_opts(min_share, intersection_floor, imbalance_flag,
                       missing_flag, min_group_size, cross, reference_path)
    result = profile(df, overrides, opts)
    if include_provenance:
        digests = [("dataset_hash", path)]
        if reference_path:
            digests.append(("reference_hash", reference_path))
        result = dict(result, provenance=build_provenance(digests, _resolve_opts(opts), overrides))
    return result


def _compare_datasets_impl(path_a, path_b, overrides=None,
                           min_share=None, intersection_floor=None,
                           imbalance_flag=None, missing_flag=None,
                           min_group_size=None, include_provenance=True):
    overrides = overrides or {}
    df_a = _read_table_or_raise(path_a)
    df_b = _read_table_or_raise(path_b)
    _check_overrides(overrides, set(df_a.columns) | set(df_b.columns))
    opts = _build_opts(min_share, intersection_floor, imbalance_flag,
                       missing_flag, min_group_size)
    profile_a = profile(df_a, overrides, opts)
    profile_b = profile(df_b, overrides, opts)
    result = compare(profile_a, profile_b, name_a=path_a, name_b=path_b)
    if include_provenance:
        provenance = build_provenance(
            [("dataset_hash_a", path_a), ("dataset_hash_b", path_b)],
            _resolve_opts(opts), overrides)
        result = dict(result, provenance=provenance)
    return result


def _proxy_hints_impl(path, overrides=None, min_share=None, min_group_size=None,
                      held_out_with=None):
    """Only min_share/min_group_size are exposed - they're the only two
    threshold knobs that feed dimension detection (profiler.py's
    _dimension()); intersection_floor/imbalance_flag/missing_flag affect
    intersections/flags, which this tool never touches.

    `held_out_with` mirrors the CLI's --proxy-hints-with: a list of
    "PATH=COLUMN" strings for testing a protected attribute that's already
    been dropped from the dataset at `path`. Parsed via proxy.py's shared
    parse_held_out_specs, so the column/row-count validation is identical to
    the CLI's.

    Returns a dict, not a bare list: the MCP SDK splits a list return value
    into one content block per element (confirmed - a 98-item result became
    98 separate content blocks, and an empty list became zero blocks, which
    is indistinguishable from an error to a caller). Wrapping in {"hints":
    [...]} keeps this tool's output shape consistent with the other two -
    always exactly one JSON object - and makes "no hints found" unambiguous.
    """
    overrides = overrides or {}
    df = _read_table_or_raise(path)
    _check_overrides(overrides, df.columns)
    opts = _build_opts(min_share=min_share, min_group_size=min_group_size)
    result = profile(df, overrides, opts)
    held_out = parse_held_out_specs(held_out_with, df, _read_table_or_raise,
                                    flag="held_out_with") if held_out_with else None
    return {"hints": compute_proxy_hints(df, result["dimensions"], held_out=held_out)}


def build_server():
    """Build the MCPServer instance with every Phase 1 tool registered."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer(
        "faircode",
        version=__version__,
        instructions=(
            "Profile a tabular dataset for demographic representation gaps, "
            "compare two datasets for representation drift, or flag columns "
            "that may be a statistical proxy for a protected attribute - all "
            "locally, no data leaves this machine. Wraps the same faircode "
            "Python API the `faircode` CLI uses."
        ),
    )

    def _as_tool_error(exc):
        return ToolError(str(exc))

    @server.tool()
    def profile_dataset(path: str, overrides: dict[str, str] | None = None,
                        cross: list[str] | None = None,
                        reference_path: str | None = None,
                        min_share: float | None = None,
                        intersection_floor: float | None = None,
                        imbalance_flag: float | None = None,
                        missing_flag: float | None = None,
                        min_group_size: int | None = None,
                        include_provenance: bool = True) -> dict:
        """Profile a tabular dataset (.csv/.tsv/.xlsx/.json/.parquet) for
        demographic representation: per-dimension imbalance/missing/skew,
        intersectional gaps, and an overall score/grade.

        `overrides` forces a column's dimension when auto-detection misses or
        mislabels it, e.g. {"gndr": "sex"}. `cross` picks two columns for the
        intersectional gap (default: the first two detected dimensions).
        `reference_path` scores against a reference baseline file (columns:
        column,group,share). The threshold args override the profiler's
        defaults (min_share=0.05, intersection_floor=0.01, imbalance_flag=3.0,
        missing_flag=0.05, min_group_size=100) when set.

        `include_provenance` (default true) attaches a provenance block -
        faircode version, a SHA-256 hash of the dataset file, and the resolved
        thresholds - so the result can be tied back to exactly what produced
        it later, without having to trust whoever ran it.
        """
        try:
            return _profile_dataset_impl(
                path, overrides, cross, reference_path, min_share,
                intersection_floor, imbalance_flag, missing_flag,
                min_group_size, include_provenance)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _as_tool_error(exc) from exc

    @server.tool()
    def compare_datasets(path_a: str, path_b: str,
                         overrides: dict[str, str] | None = None,
                         min_share: float | None = None,
                         intersection_floor: float | None = None,
                         imbalance_flag: float | None = None,
                         missing_flag: float | None = None,
                         min_group_size: int | None = None,
                         include_provenance: bool = True) -> dict:
        """Compare two tabular datasets (e.g. a training set and a production
        snapshot) for representation drift: which dimensions/groups appeared,
        disappeared, or shifted share, plus a population-stability-index-based
        drift level per dimension. `overrides` and the threshold args are
        applied identically to both datasets - see profile_dataset for what
        each one does. `include_provenance` (default true) attaches
        `dataset_hash_a`/`dataset_hash_b` alongside the resolved thresholds.
        """
        try:
            return _compare_datasets_impl(
                path_a, path_b, overrides, min_share, intersection_floor,
                imbalance_flag, missing_flag, min_group_size, include_provenance)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _as_tool_error(exc) from exc

    @server.tool()
    def proxy_hints(path: str, overrides: dict[str, str] | None = None,
                    min_share: float | None = None,
                    min_group_size: int | None = None,
                    held_out_with: list[str] | None = None) -> dict:
        """Flag pairs of detected demographic columns that are strongly
        statistically associated (chi-squared test of independence, p < 0.05)
        - a "this column may be a proxy for that protected attribute" signal.
        Returns {"hints": [...]}, most-significant pair first; an empty list
        means no pair crossed the significance threshold, not an error.

        Needs the optional 'scipy' extra (`pip install faircode[proxy]`).

        This only tests columns present in the dataset at `path` by default:
        if a protected attribute has already been dropped entirely (a common
        but naive attempt at "fixing" bias by removing the sensitive column),
        nothing here can flag a remaining column as a proxy for it unless
        `held_out_with` is given. `held_out_with` is a list of "PATH=COLUMN"
        strings (mirroring the CLI's --proxy-hints-with flag), each pointing
        at a file whose rows align 1:1 with `path` and a column to pull the
        dropped attribute's original values from. See faircode/SPEC.md
        section 3 and issue #328.
        """
        try:
            return _proxy_hints_impl(path, overrides, min_share, min_group_size, held_out_with)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _as_tool_error(exc) from exc

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
