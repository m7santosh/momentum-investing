"""Swing-trading cheat sheet copy for ETF RRG side panel."""

from __future__ import annotations

# (section title, bullet lines)
ETF_SWING_CHEAT_SHEET: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Process takeaway (weekly rebalance)",
        (
            "Sort by Change %, then keep only Rank Δ > 0",
            "Prefer names new in Top 10 Now vs falling Was → Now rank",
            "When themes rotate (e.g. IT → power/metal), favor new Leading cluster",
            "Do not average down old theme if Rank Δ is negative",
            "Improving (blue) only with large Rank Δ (e.g. +15 or more)",
            "Last week’s entry ≠ this week’s hold — recheck quadrant + Rank Δ",
        ),
    ),
    (
        "Recommended settings (3m swing)",
        (
            "Unit: Week",
            "Tail: 1–2",
            "Date: latest EOD",
            "Change % start → end: see calc line under Date",
            "Day unit: only to time entries inside the week",
        ),
    ),
    (
        "Row colors (selected Date only)",
        (
            "Green Leading — strong RS & momentum vs benchmark; hold/add leaders",
            "Blue Improving — RS recovering; watchlist / early swing",
            "Yellow Weakening — RS ok but momentum fading; trim or tighten stops",
            "Red Lagging — weak RS & momentum; avoid new longs",
        ),
    ),
    (
        "Table columns",
        (
            "Rank — best Change % over your tail window",
            "Rank Δ — move vs prior bar",
            "Price — underlying index close; trade the Ref ETF",
            "Change % — return from tail start → end (not RRG)",
        ),
    ),
    (
        "Favor for long swings",
        (
            "Leading or Improving + positive Rank Δ + top Change %",
            "In Top 10 Now with improving Was → Now",
            "RRG graph: tail drifting up-right into Leading",
        ),
    ),
    (
        "Avoid / exit",
        (
            "Weakening or Lagging with negative Rank Δ",
            "High Change % alone in Weakening (often late)",
            "Names dropping out of Top 10 Now",
        ),
    ),
    (
        "Top 10 — Was vs Now",
        (
            "Was (rank) = that line’s rank on the prior bar",
            "Now (rank) = rank at your selected Date",
            "Fresh leaders: new name climbing into Now",
        ),
    ),

)

STOCK_SWING_CHEAT_SHEET: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Process takeaway (weekly rebalance)",
        (
            "Sort by Change %, then keep only Rank Δ > 0",
            "Prefer names new in Top 10 Now vs falling Was → Now rank",
            "When sectors rotate, favor new Leading cluster",
            "Do not average down old theme if Rank Δ is negative",
            "Improving (blue) only with large Rank Δ (e.g. +15 or more)",
            "Last week’s entry ≠ this week’s hold — recheck quadrant + Rank Δ",
        ),
    ),
    (
        "Recommended settings (3m swing)",
        (
            "Unit: Week",
            "Tail: 1–2",
            "Date: latest EOD",
            "Change % start → end: see calc line under Date",
            "Day unit: only to time entries inside the week",
        ),
    ),
    (
        "Row colors (selected Date only)",
        (
            "Green Leading — strong RS & momentum vs benchmark; hold/add leaders",
            "Blue Improving — RS recovering; watchlist / early swing",
            "Yellow Weakening — RS ok but momentum fading; trim or tighten stops",
            "Red Lagging — weak RS & momentum; avoid new longs",
        ),
    ),
    (
        "Table columns",
        (
            "Rank — best Change % over your tail window",
            "Rank Δ — move vs prior bar",
            "Price — stock close vs benchmark",
            "Change % — return from tail start → end (not RRG)",
            "Industry — sector for diversification",
        ),
    ),
    (
        "Favor for long swings",
        (
            "Leading or Improving + positive Rank Δ + top Change %",
            "In Top 10 Now with improving Was → Now",
            "RRG graph: tail drifting up-right into Leading",
            "One name per industry in Recommended picks",
        ),
    ),
    (
        "Avoid / exit",
        (
            "Weakening or Lagging with negative Rank Δ",
            "High Change % alone in Weakening (often late)",
            "Names dropping out of Top 10 Now",
        ),
    ),
    (
        "Top 10 — Was vs Now",
        (
            "Was (rank) = that line’s rank on the prior bar",
            "Now (rank) = rank at your selected Date",
            "Fresh leaders: new name climbing into Now",
        ),
    ),
)
