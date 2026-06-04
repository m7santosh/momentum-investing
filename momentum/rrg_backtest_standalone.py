"""Launch standalone RRG backtest window (India or US ETFs)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from momentum.rrg_backtest_ui import launch_standalone_rrg_backtest  # noqa: E402

_GUI_CHILD_FLAG = "--gui-child"
_READY_ENV = "RRG_BACKTEST_READY_FILE"


def _pythonw_exe() -> str:
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return str(pythonw if pythonw.is_file() else exe)


def _subprocess_creationflags() -> int:
    if sys.platform == "win32":
        return subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone RRG backtest window (India or US ETFs)."
    )
    parser.add_argument(
        "--profile",
        choices=("india", "us", "stock"),
        default="india",
        help="india = NSE ETFs vs Nifty 500; us = US ETFs vs S&P 500; stock = NSE stocks",
    )
    parser.add_argument(
        "--start",
        help="Backtest start date (DD-MM-YYYY or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="Backtest end date (DD-MM-YYYY or YYYY-MM-DD)",
    )
    parser.add_argument("--top-n", type=int, default=7, dest="top_n")
    parser.add_argument(
        "--tail",
        type=int,
        help="RRG tail weeks (default: 1, same as live US/India RRG)",
    )
    parser.add_argument("--window", type=int, default=10, dest="rrg_window")
    return parser.parse_args(argv)


def _launch_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        profile=args.profile,
        rrg_window=args.rrg_window,
        tail=args.tail,
        top_n=args.top_n,
        start=args.start,
        end=args.end,
    )


def _run_gui_child(args: argparse.Namespace) -> None:
    ready_file = os.environ.get(_READY_ENV) or None
    launch_standalone_rrg_backtest(**_launch_kwargs(args), ready_file=ready_file)


def _run_console_bootstrap(args: argparse.Namespace) -> int:
    label = {"india": "India", "us": "US", "stock": "Stock"}.get(args.profile, args.profile)
    print(f"Starting {label} RRG backtest window...")
    ready_path = Path(tempfile.mktemp(suffix=".rrg_ready"))
    env = os.environ.copy()
    env[_READY_ENV] = str(ready_path)

    cmd = [str(Path(__file__).resolve()), _GUI_CHILD_FLAG]
    if args.profile not in ("india",):
        cmd.extend(["--profile", args.profile])
    if args.start:
        cmd.extend(["--start", args.start])
    if args.end:
        cmd.extend(["--end", args.end])
    if args.top_n != 7:
        cmd.extend(["--top-n", str(args.top_n)])
    if args.tail is not None:
        cmd.extend(["--tail", str(args.tail)])
    if args.rrg_window != 10:
        cmd.extend(["--window", str(args.rrg_window)])

    proc = subprocess.Popen(
        [_pythonw_exe(), *cmd],
        env=env,
        cwd=str(_PROJECT_ROOT),
        creationflags=_subprocess_creationflags(),
        close_fds=True,
    )

    deadline = time.monotonic() + 120.0
    exit_code = 0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print("Error: backtest exited before the window opened.")
                exit_code = 1
                break
            if ready_path.is_file() and ready_path.read_text(encoding="utf-8").strip():
                print("Backtest window open.")
                break
            time.sleep(0.05)
        else:
            print("Error: timed out waiting for backtest window.")
            exit_code = 1
    finally:
        ready_path.unlink(missing_ok=True)

    return exit_code


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    gui_child = _GUI_CHILD_FLAG in raw
    if gui_child:
        raw.remove(_GUI_CHILD_FLAG)
    args = _parse_args(raw)
    if gui_child:
        _run_gui_child(args)
        return 0
    return _run_console_bootstrap(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
