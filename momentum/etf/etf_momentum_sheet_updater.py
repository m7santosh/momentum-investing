"""
ETF Momentum screen → Google Sheets (headless, for GitHub Actions).

Run:
    python momentum/etf/etf_momentum_sheet_updater.py

Outputs:
  - Abs Momentum
  - RS Blended
  - RS Adaptive
  - Top Picks
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from momentum.etf.etf_momentum_engine import fetch_etf_momentum_snapshot  # noqa: E402
from momentum.etf.etf_momentum_recommendations import recommendations_dataframe  # noqa: E402
from utils.nse_bhavcopy import today_ist  # noqa: E402


def setup_gsheet_client():
    """Setup Google Sheets client using GCP credentials from environment."""
    creds_json = os.environ.get('ETF_MOMENTUM_CREDENTIAL')
    if not creds_json:
        print("ERROR: ETF_MOMENTUM_CREDENTIAL secret missing!")
        sys.exit(1)

    creds_dict = json.loads(creds_json)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)


def update_sheet(ws, title: str, df: pd.DataFrame) -> None:
    """Update worksheet with dataframe data."""
    if df is None or df.empty:
        print(f"⚠️  {title}: No data to write")
        return

    # Bulk write all data in a single request to avoid quota throttling.
    headers = list(df.columns)
    values = [headers]
    for _, row in df.iterrows():
        values.append([
            row[col] if not pd.isna(row[col]) else ""
            for col in headers
        ])

    ws.clear()
    ws.update("A1", values)

    print(f"✓ {title}: Updated with {len(df)} rows")


def main():
    client = setup_gsheet_client()

    # Get your Google Sheet ID from environment or hardcode
    # spreadsheet_id = os.environ.get(
    #     'ETF_MOMENTUM_CREDENTIAL',
    #     "1SW72fnMr-I0wjJbNMxTFguBrw-vpRGpxXXO-SiFBO_8"  # Replace with your sheet ID
    # )
    spreadsheet_id = "1SW72fnMr-I0wjJbNMxTFguBrw-vpRGpxXXO-SiFBO_8"

    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except Exception as e:
        print(f"ERROR: Could not open Google Sheet: {e}")
        sys.exit(1)

    # Fetch ETF momentum snapshot
    print("Fetching ETF momentum rankings…")
    snapshot = fetch_etf_momentum_snapshot()

    print(f"Run date: {snapshot.run_date}")
    print(f"Market regime: {snapshot.market_regime}")

    # Ensure worksheets exist (create if needed)
    sheet_names = [ws.title for ws in spreadsheet.worksheets()]

    def get_or_create_worksheet(name: str):
        if name in sheet_names:
            return spreadsheet.worksheet(name)
        else:
            return spreadsheet.add_worksheet(title=name, rows=10000, cols=50)

    # Update each sheet
    ws_abs = get_or_create_worksheet("ETF Abs Momentum")
    update_sheet(ws_abs, "Abs Momentum", snapshot.abs_momentum)

    ws_rs = get_or_create_worksheet("ETF RS Blended")
    update_sheet(ws_rs, "RS Blended", snapshot.rs_blended)

    ws_adaptive = get_or_create_worksheet("ETF RS Adaptive")
    update_sheet(ws_adaptive, "RS Adaptive", snapshot.rs_adaptive)

    # Update picks
    picks_df = recommendations_dataframe(
        snapshot.abs_momentum,
        snapshot.rs_blended,
        snapshot.rs_adaptive,
    )
    ws_picks = get_or_create_worksheet("ETF Top Picks")
    update_sheet(ws_picks, "Top Picks", picks_df)

    # Update summary
    ist = timezone(timedelta(hours=5, minutes=30))
    summary_data = [
        ["Metric", "Value"],
        ["Run Date", snapshot.run_date],
        ["Market Regime", snapshot.market_regime],
        ["Abs Momentum Count", len(snapshot.abs_momentum)],
        ["RS Blended Count", len(snapshot.rs_blended)],
        ["RS Adaptive Count", len(snapshot.rs_adaptive)],
        ["ETFs Ranked (Adaptive)", snapshot.etfs_ranked_adaptive],
        ["Updated At", datetime.now(tz=ist).strftime("%Y-%m-%d %H:%M:%S IST")],
    ]
    ws_summary = get_or_create_worksheet("Summary")
    ws_summary.clear()
    ws_summary.update("A1", summary_data)
    print("✓ Summary: Updated")

    print("\n✅ ETF momentum screen updated successfully!")


if __name__ == "__main__":
    main()
