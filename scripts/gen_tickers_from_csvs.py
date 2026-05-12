"""Build a Python `tickers = [...]` fragment from index CSVs.

Expected columns include **Symbol**, **Industry**, and **Company Name** or **Company** (used as a
fallback for the ``industry`` field in the output dict when **Industry** is blank). **Series** is
ignored unless you pass ``--eq-only`` (then only ``Series == EQ`` rows are kept).

Merge order: every ``--source`` left-to-right, then every data line in ``--sources-file`` top to
bottom. On duplicate symbols, the first occurrence wins. Symbols whose name starts with
``DUMMY`` (case-insensitive) are skipped by default.

Provide at least one of ``-o`` / ``--out`` (use ``-`` for stdout) and/or ``--apply-to``.
To patch a module from a saved fragment (no CSV run), use ``--from-source FILE`` or rely on the
default ``scripts/_tickers_fragment.txt`` when that file exists and you pass ``--apply-to`` without
``--source`` / ``--sources-file``.
Section comments ``# --- BUCKET: filename.csv ---`` are inserted before each source file's
symbols (in output order) so plain-text fragments are easy to scan; a blank line separates one
bucket block from the next. They are valid inside the
Python ``tickers = [ ... ]`` list when using ``--apply-to``.

Examples (single line; works in PowerShell, cmd, and bash)::

  python scripts/gen_tickers_from_csvs.py --source Nifty50 C:/path/ind_nifty50list.csv --source Next50 C:/path/ind_niftynext50list.csv -o scripts/_tickers_fragment.txt

PowerShell only: use a backtick (not backslash) to break lines::

  python scripts/gen_tickers_from_csvs.py `
  --source Nifty50 "C:/Users/SantoshMandal/Downloads/ind_nifty50list.csv" `
  --source Next50 "C:/Users/SantoshMandal/Downloads/ind_niftynext50list.csv" `
  --source Midcap "C:/Users/SantoshMandal/Downloads/ind_niftymidcap150list (1).csv" `
  --source Smallcap "C:/Users/SantoshMandal/Downloads/ind_niftysmallcap250list (2).csv" `
  -o scripts/_tickers_fragment.txt

  python scripts/gen_tickers_from_csvs.py --apply-to momentum/stock/quality_momentum_rs_lv_n500.py

  # same as above when scripts/_tickers_fragment.txt exists; or set path explicitly:
  python scripts/gen_tickers_from_csvs.py --from-source scripts/tickers_list.txt --apply-to momentum/stock/quality_momentum_rs_lv_n500.py

Create ``scripts/my_sources.txt`` from ``scripts/my_sources.example.txt`` (uncomment the CSV lines and set paths), or pass any path to ``--sources-file``. Each active line is ``BUCKET,csv_path``.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent


EXAMPLE_SOURCES_FILE = SCRIPT_PATH.parent / "my_sources.example.txt"


def _sources_file_is_ticker_fragment_exit(path: Path) -> None:
    sys.exit(
        f"{path} looks like ticker output (dict lines from -o/--out), not a --sources-file.\n"
        f"  --sources-file must list NSE CSV paths, one per line: BUCKET,path/to/index.csv\n"
        f"  Template: {EXAMPLE_SOURCES_FILE}"
    )


def load_sources_file(path: Path) -> list[tuple[str, Path]]:
    if not path.is_file():
        msg = (
            f"Sources file not found: {path}\n"
            f"  Current directory: {Path.cwd()}\n"
            f"  Use an absolute path, or create the file under cwd, or a path relative to cwd."
        )
        if EXAMPLE_SOURCES_FILE.is_file():
            msg += (
                f"\n  Template (copy, uncomment lines, set paths): {EXAMPLE_SOURCES_FILE}\n"
                f"  Example: --sources-file scripts/my_sources.txt"
            )
        sys.exit(msg)

    raw = path.read_text(encoding="utf-8")
    head = raw[:250000]
    if '{"symbol"' in head and '"marketcap"' in head:
        _sources_file_is_ticker_fragment_exit(path)

    out: list[tuple[str, Path]] = []
    for lineno, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            sys.exit(f"{path}:{lineno}: expected 'BUCKET,csv_path', got: {raw_line!r}")
        bucket, _, rest = line.partition(",")
        bucket = bucket.strip()
        if "{" in bucket or '"symbol"' in bucket.lower():
            _sources_file_is_ticker_fragment_exit(path)
        p = Path(rest.strip()).expanduser()
        if not bucket:
            sys.exit(f"{path}:{lineno}: empty bucket label")
        out.append((bucket, p))
    return out


def _resolve_path(p: Path) -> Path:
    p = p.expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def _csv_missing_exit(bucket: str, p: Path) -> None:
    msg = f"CSV not found for bucket {bucket!r}: {p}"
    norm = str(p).replace("\\", "/").lower()
    if "users/me/" in norm:
        msg += (
            "\n  Hint: that path uses the doc placeholder 'Users/me'. "
            "Use your real Windows profile, e.g. C:/Users/<YourAccount>/Downloads/....csv"
        )
    elif "/path/" in norm:
        msg += "\n  Hint: replace C:/path/... with the real folder where your CSV files are."
    sys.exit(msg)


def read_fragment_lines(path: Path) -> list[str]:
    """Load lines written by -o/--out (header + section comments + dict lines) for --apply-to."""
    path = path.resolve()
    if not path.is_file():
        sys.exit(f"Fragment file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    lines = [ln.rstrip("\r\n") for ln in raw.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        sys.exit(f"Empty fragment: {path}")
    if not any('{"symbol"' in ln for ln in lines):
        sys.exit(
            f"{path}: no ticker dict lines (missing '{{\"symbol\"' …). "
            "This file is not a valid fragment from -o/--out."
        )
    if not lines[0].lstrip().startswith("#"):
        nd = sum(1 for ln in lines if '{"symbol"' in ln)
        lines.insert(0, f"# total symbols (from fragment): {nd}")
    return lines


def _clean_cell(val: object) -> str:
    return (str(val) if val is not None else "").strip().replace('"', "'")


def row_industry_label(
    row: dict[str, str | None],
    industry_col: str,
    company_fallbacks: tuple[str, ...],
) -> str:
    """Industry column first; then each company fallback column in order."""
    v = _clean_cell(row.get(industry_col, ""))
    if v:
        return v
    for col in company_fallbacks:
        v = _clean_cell(row.get(col, ""))
        if v:
            return v
    return ""


def collect_sources(
    cli_pairs: list[list[str]] | None,
    sources_file: Path | None,
) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if cli_pairs:
        for pair in cli_pairs:
            bucket, csv_s = pair[0].strip(), pair[1].strip()
            if not bucket:
                sys.exit("--source: bucket label must be non-empty")
            out.append((bucket, Path(csv_s)))
    if sources_file is not None:
        out.extend(load_sources_file(sources_file))
    if not out:
        sys.exit("Provide --source BUCKET CSV (repeat) and/or --sources-file PATH")

    resolved: list[tuple[str, Path]] = []
    for bucket, p in out:
        p = _resolve_path(p)
        if not p.is_file():
            _csv_missing_exit(bucket, p)
        resolved.append((bucket, p))
    return resolved


def build_ticker_lines(
    sources: list[tuple[str, Path]],
    *,
    symbol_col: str,
    industry_col: str,
    company_fallbacks: tuple[str, ...],
    series_col: str | None,
    require_series: str | None,
    skip_symbol_prefix: str | None,
    suffix: str,
    encoding: str,
) -> list[str]:
    rows: list[tuple[str, str, str, Path]] = []
    seen: set[str] = set()
    for bucket, path in sources:
        with path.open(encoding=encoding, newline="") as f:
            for row in csv.DictReader(f):
                sym = (row.get(symbol_col) or "").strip()
                if not sym:
                    continue
                if skip_symbol_prefix and sym.upper().startswith(skip_symbol_prefix.upper()):
                    continue
                if series_col is not None and require_series is not None:
                    ser = (row.get(series_col) or "").strip()
                    if ser != require_series:
                        continue
                ind = row_industry_label(row, industry_col, company_fallbacks)
                if sym in seen:
                    continue
                seen.add(sym)
                yahoo = f"{sym}{suffix}" if suffix else sym
                rows.append((yahoo, ind, bucket, path))

    lines = [f"# total symbols (deduped): {len(rows)}"]
    prev_key: tuple[str, Path] | None = None
    for yahoo, industry, bucket, src_path in rows:
        key = (bucket, src_path.resolve())
        if key != prev_key:
            if prev_key is not None:
                lines.append("")
            lines.append(f"    # --- {bucket}: {src_path.name} ---")
            prev_key = key
        ind_esc = industry.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'    {{"symbol": "{yahoo}", "industry": "{ind_esc}", "marketcap": "{bucket}"}},'
        )
    return lines


def write_out(lines: list[str], out: Path | None, to_stdout: bool) -> None:
    text = "\n".join(lines) + "\n"
    n_tickers = sum(1 for ln in lines if '{"symbol"' in ln)
    if to_stdout:
        sys.stdout.write(text)
        print(f"(stdout: {n_tickers} ticker lines)", file=sys.stderr)
        return
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {n_tickers} entries -> {out}", file=sys.stderr)


def splice_tickers_into_module(
    lines: list[str],
    py_path: Path,
    *,
    start_marker: str,
    end_marker: str,
    regen_argv: list[str],
    sources: list[tuple[str, Path]] | None,
    fragment_path: Path | None,
) -> None:
    py_path = py_path.resolve()
    if not py_path.is_file():
        sys.exit(f"Not a file: {py_path}")

    try:
        regen_arg = py_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        regen_arg = str(py_path)

    try:
        script_rel = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        script_rel = SCRIPT_PATH.name

    regen_cmd = shlex.join(["python", script_rel, *regen_argv])
    if fragment_path is not None:
        try:
            src_desc = fragment_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            src_desc = str(fragment_path)
    else:
        if not sources:
            sys.exit("internal: splice needs sources or fragment_path")
        src_desc = ", ".join(f"{b}({p.name})" for b, p in sources)
    nd = sum(1 for ln in lines[1:] if '{"symbol"' in ln)
    entries = "\n".join(lines[1:])
    new_block = f"""# --- Ticker universe (from {src_desc}).
#     Ticker dict lines in list: {nd}.
#     Regenerate: {regen_cmd}
tickers = [
{entries}
]"""
    text = py_path.read_text(encoding="utf-8")
    i0 = text.index(start_marker)
    post = text.index(end_marker, i0)
    py_path.write_text(text[:i0] + new_block + text[post:], encoding="utf-8", newline="\n")
    print(f"Updated {py_path}", file=sys.stderr)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge index CSVs into Python ticker dict lines (default: Yahoo NSE .NS suffix).",
    )
    p.add_argument(
        "--source",
        nargs=2,
        metavar=("BUCKET", "CSV"),
        action="append",
        default=None,
        help="Bucket label (stored as marketcap) and CSV path. Repeat in merge order.",
    )
    p.add_argument(
        "--sources-file",
        type=Path,
        metavar="FILE",
        help="Each non-comment line: BUCKET,csv_path. See scripts/my_sources.example.txt.",
    )
    p.add_argument(
        "-o",
        "--out",
        type=str,
        metavar="PATH",
        help="Write fragment to this path, or '-' for stdout.",
    )
    p.add_argument(
        "--from-source",
        type=Path,
        metavar="FILE",
        help="Use this pre-built fragment (from -o/--out) for --apply-to and/or -o instead of CSVs. "
        "If --apply-to is set without CSVs and without this flag, scripts/_tickers_fragment.txt is used when present.",
    )
    p.add_argument(
        "--apply-to",
        type=Path,
        metavar="PYFILE",
        help="Repo-relative or absolute .py: replace block from --start-marker through tickers list.",
    )
    p.add_argument(
        "--start-marker",
        default="# --- Ticker universe",
        help="Start of region to replace in --apply-to target (default: %(default)s — matches generated blocks).",
    )
    p.add_argument(
        "--end-marker",
        default="\n\n# --- Helper Functions ---",
        help="Text immediately after the tickers closing bracket (default: newline + Helper comment).",
    )
    p.add_argument(
        "--symbol-column",
        default="Symbol",
        help="CSV column for symbol (default: %(default)s).",
    )
    p.add_argument(
        "--industry-column",
        default="Industry",
        help="CSV column for industry (default: %(default)s).",
    )
    p.add_argument(
        "--company-column",
        action="append",
        metavar="NAME",
        dest="company_columns",
        help="Extra column(s) tried in order after Industry when Industry is blank. "
        "Default fallbacks: Company Name, Company. Repeat flag to set order.",
    )
    p.add_argument(
        "--eq-only",
        action="store_true",
        help="Only include rows where Series (see --series-column) equals EQ.",
    )
    p.add_argument(
        "--series-column",
        default="Series",
        help="Series column name when --eq-only is set (default: %(default)s).",
    )
    p.add_argument(
        "--require-series",
        default="EQ",
        help="Value required in series column when --eq-only (default: %(default)s).",
    )
    p.add_argument(
        "--skip-symbol-prefix",
        default="DUMMY",
        help="Skip symbols with this prefix (case-insensitive); empty to disable (default: %(default)s).",
    )
    p.add_argument(
        "--suffix",
        default=".NS",
        help="Suffix after symbol (default: %(default)s).",
    )
    p.add_argument(
        "--bare-symbol",
        action="store_true",
        help="Append no suffix (overrides --suffix).",
    )
    p.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding (default: %(default)s).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv
    args = _parse_args(argv[1:])

    if not args.out and args.apply_to is None:
        sys.exit("Specify at least one of -o/--out PATH (or -o -) and/or --apply-to PYFILE")

    has_csv = bool(args.source) or (args.sources_file is not None)

    frag_path: Path | None = None
    if args.from_source is not None:
        frag_path = _resolve_path(args.from_source)
    elif args.apply_to is not None and not has_csv:
        cand = REPO_ROOT / "scripts" / "_tickers_fragment.txt"
        if cand.is_file():
            frag_path = cand.resolve()

    if frag_path is not None and has_csv:
        sys.exit("Use either CSV inputs (--source / --sources-file) or --from-source / default fragment, not both.")

    sources: list[tuple[str, Path]] = []
    lines: list[str]

    if has_csv:
        sf = _resolve_path(args.sources_file) if args.sources_file else None
        sources = collect_sources(args.source, sf)

        if args.eq_only:
            series_col = (args.series_column or "").strip() or None
            req_series = (args.require_series or "").strip() or None
            if not series_col:
                sys.exit("--eq-only requires a non-empty --series-column (default: Series)")
            if not req_series:
                req_series = "EQ"
        else:
            series_col = None
            req_series = None

        company_fallbacks: tuple[str, ...] = (
            tuple(args.company_columns) if args.company_columns else ("Company Name", "Company")
        )

        skip_pfx = (args.skip_symbol_prefix or "").strip() or None
        suffix = "" if args.bare_symbol else (args.suffix or "")

        lines = build_ticker_lines(
            sources,
            symbol_col=args.symbol_column,
            industry_col=args.industry_column,
            company_fallbacks=company_fallbacks,
            series_col=series_col,
            require_series=req_series,
            skip_symbol_prefix=skip_pfx,
            suffix=suffix,
            encoding=args.encoding,
        )
    elif frag_path is not None:
        lines = read_fragment_lines(frag_path)
    else:
        sys.exit(
            "Provide CSV inputs (--source / --sources-file), "
            "or --from-source PATH, "
            "or create scripts/_tickers_fragment.txt and use --apply-to without CSVs. "
            "Use -o/--out to write a new fragment from CSVs."
        )

    if args.out:
        to_stdout = args.out.strip() == "-"
        out_path = None if to_stdout else _resolve_path(Path(args.out))
        write_out(lines, out_path, to_stdout)

    if args.apply_to is not None:
        target = args.apply_to.expanduser()
        if not target.is_absolute():
            target = (REPO_ROOT / target).resolve()
        else:
            target = target.resolve()
        regen_argv = list(argv[1:])
        splice_tickers_into_module(
            lines,
            target,
            start_marker=args.start_marker,
            end_marker=args.end_marker,
            regen_argv=regen_argv,
            sources=sources if has_csv else None,
            fragment_path=frag_path if not has_csv else None,
        )


if __name__ == "__main__":
    main()
