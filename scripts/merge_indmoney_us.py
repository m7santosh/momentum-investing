"""One-off: merge us_indmoney.py into us.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentum.etf.universes import us as us_core
from momentum.etf.universes.us_indmoney import INDMONEY_CATEGORIES, INDMONEY_EXCLUDED

OUT = ROOT / "momentum" / "etf" / "universes" / "us.py"

excluded = {t for pairs in INDMONEY_EXCLUDED.values() for t, _ in pairs}
indmoney_names: dict[str, str] = {}
for pairs in INDMONEY_CATEGORIES.values():
    for t, name in pairs:
        if t not in excluded:
            indmoney_names[t] = name

current = [t for t in us_core.tickers if t not in excluded]
current_set = set(current)
labels: dict[str, str] = {
    t: indmoney_names.get(t, us_core.ETF_LABELS[t]) for t in current
}

new_sections: list[tuple[str, list[str]]] = []
seen_new: set[str] = set()
for cat, pairs in INDMONEY_CATEGORIES.items():
    block: list[str] = []
    for t, _ in pairs:
        if t not in excluded and t not in current_set and t not in seen_new:
            block.append(t)
            seen_new.add(t)
            labels[t] = indmoney_names[t]
    if block:
        new_sections.append((cat, block))

tickers = list(current)
for _, block in new_sections:
    tickers.extend(block)

assert len(tickers) == len(set(tickers)) == 268
assert all(t in labels for t in tickers)

sections = [
    (
        "US broad / style",
        [
            "SPY", "VOO", "IVV", "SPYM", "VTI", "QQQ", "QQQM", "IWM", "DIA", "RSP",
            "MDY", "SCHX", "XLG", "MGK", "VUG", "VTV", "IWF", "IWD",
        ],
    ),
    (
        "GICS sectors (SPDR)",
        ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"],
    ),
    (
        "Sector / industry",
        [
            "VGT", "VHT", "VDE", "VFH", "VIS", "VCR", "VDC", "VPU", "VAW", "SOXX", "SMH",
            "XSD", "XBI", "IBB", "XHB", "ITB", "XRT", "KBE", "KRE", "XOP", "OIH", "XES",
            "IYT", "IGV", "XSW", "ITA", "XAR", "XPH", "XHE", "KIE", "JETS", "XME", "PICK",
            "PAVE", "IGF",
        ],
    ),
    (
        "Thematic / innovation",
        [
            "ARKK", "ARKG", "ARKW", "SKYY", "CLOU", "CIBR", "HACK", "AIQ", "BAI", "DRAM",
            "THRO", "ROBO", "BOTZ", "ICLN", "QCLN", "TAN", "GRID", "LIT", "URA", "URNM",
            "GUNR", "MOO", "COPX", "REMX", "DXYZ",
        ],
    ),
    (
        "International / country",
        [
            "VXUS", "VEA", "EFA", "EEM", "IEMG", "VWO", "EZU", "MCHI", "FXI", "ASHR",
            "CQQQ", "KWEB", "EMXC", "INDA", "EPI", "EWJ", "EWY", "EWT", "EWA", "EWC",
            "EWG", "EWU", "EWUS", "EWP", "EWH", "EWS", "EWM", "EIS", "EZA", "EIDO", "THD",
            "EPOL", "EWW", "VNM", "ECH", "EWL", "EWN", "EWD", "EWI", "EWZ", "ARGT", "TUR",
            "GREK", "KSA", "FM", "RSX",
        ],
    ),
    ("Real estate", ["VNQ", "IYR"]),
    (
        "Commodities / metals",
        [
            "GLD", "SLV", "GDX", "GDXJ", "SIL", "GSG", "DBC", "USO", "UNG", "DBA", "WEAT",
            "CORN", "CPER", "PALL", "PPLT",
        ],
    ),
    ("Crypto", ["BITO", "IBIT", "WGMI", "ETHA"]),
    ("Bonds / cash", ["SGOV", "BND", "TLT", "HYG"]),
    ("Factor / dividend", ["MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SCHD"]),
    ("FX", ["UUP", "FXE", "FXY", "FXB"]),
]

section_map: dict[str, str] = {}
for sec_name, sec_syms in sections:
    for s in sec_syms:
        section_map[s] = sec_name

lines: list[str] = [
    '"""US-listed ETFs — single source of truth for all US ETF tools.',
    "",
    "Edit ``tickers`` and ``ETF_LABELS`` here only. Merged with INDmoney",
    "category screenshots (us_indmoney.py). Used by RRG, momentum, backtest.",
    '"""',
    "",
    'KEY = "us"',
    'LABEL = "US ETFs"',
    'DESCRIPTION = "US sector, thematic, and commodity ETFs on Yahoo Finance"',
    'BENCHMARK_YAHOO = "^GSPC"',
    "",
    "DEFAULT_VISIBLE = {",
]
for t in us_core.DEFAULT_VISIBLE:
    lines.append(f'    "{t}",')
lines.append("}")
lines.append("")
lines.append("tickers = [")


def emit_block(comment: str, syms: list[str]) -> None:
    if not syms:
        return
    lines.append(f"    # --- {comment} ---")
    for s in syms:
        lines.append(f"    '{s}',")


for sec_name, _ in sections:
    present = [t for t in current if section_map.get(t) == sec_name]
    emit_block(sec_name, present)

orphan = [t for t in current if t not in section_map]
if orphan:
    emit_block("Other (legacy)", orphan)

for cat, block in new_sections:
    emit_block(f"INDmoney: {cat}", block)

lines.append("]")
lines.append("")
lines.append("ETF_LABELS = {")
for t in tickers:
    label = labels[t].replace('"', "'")
    lines.append(f'    "{t}": "{label}",')
lines.append("}")
lines.append("")
lines.append("# --- end ticker universe ---")
lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {len(tickers)} tickers to {OUT}")
