"""Quick diagnostic to check available NSE indices and data loading."""

from datetime import date, timedelta
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.nse_bhavcopy import list_nse_index_names, fetch_index_ohlc_history

print("📊 Available NSE Indices:")
print("=" * 80)

indices = list_nse_index_names()
print(f"Total indices: {len(indices)}\n")

# Show first 20
for i, idx in enumerate(indices[:20], 1):
    print(f"  {i:2d}. {idx}")

if len(indices) > 20:
    print(f"  ... and {len(indices) - 20} more")

# Try loading data for one index
print("\n" + "=" * 80)
print("Testing data load for first 3 indices...")
print("=" * 80 + "\n")

test_date_end = date.today()
test_date_start = test_date_end - timedelta(days=365)

for idx in indices[:3]:
    print(f"Loading {idx}...", end=" ", flush=True)
    try:
        df = fetch_index_ohlc_history(idx, test_date_start, test_date_end, quiet=True)
        if df is not None and not df.empty:
            print(f"✓ ({len(df)} bars)")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Date range: {df.index[0].date()} to {df.index[-1].date()}")
        else:
            print("❌ Empty data")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 80)
