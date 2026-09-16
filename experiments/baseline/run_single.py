"""Run the SafeNav-VLM direct baseline on one image."""

import argparse
import re
import time
from pathlib import Path

from PIL import Image

from safenav_vlm.models.qwen_vlm import QwenVLM
from safenav_vlm.prompts.safety_prompt import build_safety_prompt

VALID_DECISIONS = {"SAFE", "UNSAFE", "UNCERTAIN"}


def parse_response(response: str) -> tuple[str, str]:
    """Extract decision and reason from the VLM response."""

    decision_match = re.search(
        r"Decision:\s*(SAFE|UNSAFE|UNCERTAIN)",
        response,
        re.IGNORECASE,
    )

    reason_match = re.search(
        r"Reason:\s*(.+)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    decision = (
        decision_match.group(1).upper()
        if decision_match
        else "PARSE_ERROR"
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else "No reason parsed."
    )

    return decision, reason


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SafeNav-VLM direct baseline."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--question",
        default="Can I safely walk forward?",
        help="Navigation question to ask the VLM.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        width, height = image.size

    prompt = build_safety_prompt(args.question)

    vlm = QwenVLM()

    start = time.perf_counter()

    response = vlm.generate_response(
        image_path=str(image_path),
        prompt=prompt,
    )

    elapsed = time.perf_counter() - start

    decision, reason = parse_response(response)

    print("\n--- SafeNav-VLM Direct Baseline ---")
    print(f"Image: {image_path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"Question: {args.question}")
    print(f"Decision: {decision}")
    print(f"Reason: {reason}")
    print(f"Inference time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
