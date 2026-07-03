"""`uv run evalkit run-all` — the one-command leaderboard release (SPEC_M2 §8)."""

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(prog="evalkit")
    sub = parser.add_subparsers(dest="command", required=True)
    run_all_p = sub.add_parser("run-all", help="regenerate the full evaluation")
    run_all_p.add_argument(
        "--cold",
        action="store_true",
        help="ignore the artifact cache and refit everything (documented < 60 min)",
    )
    args = parser.parse_args()
    if args.command == "run-all":
        from evalkit.run import run_all

        t0 = time.monotonic()
        run_all(cold=args.cold)
        print(
            f"run-all completed in {time.monotonic() - t0:.0f}s "
            f"({'cold' if args.cold else 'cached where valid'})"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
