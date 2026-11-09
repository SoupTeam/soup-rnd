"""Offline analysis of the E2 OLMoE cache series (olmoe_cache/PROTOCOL.txt 4-5). CPU only.

  python benchmarks/e2/olmoe_cache/analyze_olmoe.py SESSION_DIR [--out NEW.json]

The decision logic is hybrid_cache/analyze_speed.py with L = 16, K = 4 and the O16 template.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hybrid_cache"))
import analyze_speed as speed  # noqa: E402


def _matches_o16(arm_dir: Path) -> bool:
    text = (arm_dir / "config.yaml").read_text(encoding="utf-8")
    text = re.sub(r"(?m)^base: .*$", "base: __BASE__", text)
    text = re.sub(r"(?m)^output: .*$", "output: __OUTPUT__", text)
    return text == (HERE / "configs" / "O16.yaml").read_text(encoding="utf-8")


SETUP = speed.Setup(layers=16, keep_top=4, template_check=_matches_o16, template_name="O16")


def analyze(session: Path) -> dict:
    result = speed.analyze(session, SETUP)
    result["schema"] = "e2-olmoe-cache-analysis-v1"
    result["protocol"] = "benchmarks/e2/olmoe_cache/PROTOCOL.txt"
    result["not_claimed"] = [
        "quality", "other K, models, data, epochs or GPUs", "MoE in general",
        "router load-balancing loss (off in every arm)",
    ]
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(analyze(args.session), indent=2) + "\n"
    if args.out:
        with open(args.out, "x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
