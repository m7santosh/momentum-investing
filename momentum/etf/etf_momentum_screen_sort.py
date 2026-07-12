"""Click-to-sort helpers for ETF momentum Treeview tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

SORTABLE_ETF_MOMENTUM_COLUMNS = frozenset(
    {
        "Position",
        "Above_9EMA_Since",
        "Pct_Above_9EMA",
        "Return_1W",
        "Return_2W",
        "Return_1M",
        "Return_3M",
    }
)

_ASCENDING_FIRST_CLICK_COLUMNS = frozenset({"Position", "Above_9EMA_Since"})


@dataclass
class TreeSortState:
    sort_keys: list[tuple[str, bool]] = field(default_factory=list)

    @property
    def column(self) -> str | None:
        return self.sort_keys[0][0] if self.sort_keys else None

    @property
    def ascending(self) -> bool:
        return self.sort_keys[0][1] if self.sort_keys else True


def _default_ascending(column: str) -> bool:
    return column in _ASCENDING_FIRST_CLICK_COLUMNS


def update_sort_keys_on_click(
    sort_keys: list[tuple[str, bool]],
    column: str,
) -> list[tuple[str, bool]]:
    """Append a new sort level, or toggle direction if that column is already active."""
    existing_idx = next(
        (index for index, (col, _) in enumerate(sort_keys) if col == column),
        None,
    )
    if existing_idx is not None:
        col, ascending = sort_keys[existing_idx]
        updated = list(sort_keys)
        updated[existing_idx] = (col, not ascending)
        return updated
    return sort_keys + [(column, _default_ascending(column))]


def sort_etf_momentum_df(
    df: pd.DataFrame,
    column: str,
    *,
    ascending: bool,
) -> pd.DataFrame:
    return sort_etf_momentum_df_multi(df, [(column, ascending)])


def sort_etf_momentum_df_multi(
    df: pd.DataFrame,
    sort_keys: list[tuple[str, bool]],
) -> pd.DataFrame:
    if df.empty or not sort_keys:
        return df

    valid_keys = [(col, asc) for col, asc in sort_keys if col in df.columns]
    if not valid_keys:
        return df

    work = df.copy()
    by: list[str] = []
    ascending: list[bool] = []
    temp_cols: list[str] = []

    for col, asc in valid_keys:
        if col == "Above_9EMA_Since":
            key_col = f"_sort_{col}"
            work[key_col] = pd.to_datetime(work[col], errors="coerce", dayfirst=True)
            by.append(key_col)
            temp_cols.append(key_col)
        else:
            by.append(col)
        ascending.append(asc)

    out = work.sort_values(by=by, ascending=ascending, na_position="last")
    if temp_cols:
        out = out.drop(columns=temp_cols)
    return out.reset_index(drop=True)


def sort_heading_text(base: str, column: str, state: TreeSortState) -> str:
    for index, (col, ascending) in enumerate(state.sort_keys):
        if col != column:
            continue
        arrow = "↑" if ascending else "↓"
        if len(state.sort_keys) > 1:
            return f"{base} {arrow}{index + 1}"
        return f"{base} {arrow}"
    return base


def reset_tree_sort_state(state: TreeSortState) -> None:
    state.sort_keys.clear()


def update_etf_momentum_tree_headings(
    tree,
    headings: tuple[str, ...],
    labels: dict[str, str] | None,
    state: TreeSortState,
) -> None:
    label_map = labels or {}
    for col_id, heading in zip(tree["columns"], headings, strict=True):
        base = label_map.get(heading, heading)
        tree.heading(col_id, text=sort_heading_text(base, heading, state))


def wire_etf_momentum_tree_sort(
    tree,
    headings: tuple[str, ...],
    labels: dict[str, str] | None,
    state: TreeSortState,
    on_sorted: Callable[[], None],
) -> None:
    """Bind header clicks for sortable columns; call ``on_sorted`` after toggle."""
    label_map = labels or {}
    col_ids = tree["columns"]

    def _refresh_headings() -> None:
        for col_id, heading in zip(col_ids, headings, strict=True):
            base = label_map.get(heading, heading)
            text = sort_heading_text(base, heading, state)
            if heading in SORTABLE_ETF_MOMENTUM_COLUMNS:
                tree.heading(col_id, text=text, command=lambda h=heading: _on_click(h))
            else:
                tree.heading(col_id, text=text)

    def _on_click(column: str) -> None:
        if column not in SORTABLE_ETF_MOMENTUM_COLUMNS:
            return

        state.sort_keys = update_sort_keys_on_click(state.sort_keys, column)

        _refresh_headings()
        on_sorted()

    _refresh_headings()


def apply_tree_sort(df: pd.DataFrame, state: TreeSortState) -> pd.DataFrame:
    if not state.sort_keys:
        return df
    return sort_etf_momentum_df_multi(df, state.sort_keys)
