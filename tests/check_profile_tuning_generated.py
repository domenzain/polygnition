#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--inc", required=True)
    args = parser.parse_args()
    generated = subprocess.run(
        [args.python, args.script], check=True, text=True, capture_output=True
    ).stdout
    expected = Path(args.inc).read_text()
    if generated != expected:
        sys.stderr.write("profile tuning include is stale; regenerate with benchmark/generate_profile_tuning.py\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
