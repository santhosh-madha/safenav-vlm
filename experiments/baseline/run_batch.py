"""Run the SafeNav-VLM direct baseline on a folder of images."""

import argparse
import csv
import re
import time
from pathlib import Path

from PIL import Image

from safenav_vlm.models.qwen_vlm import QwenVLM
from safenav_vlm.prompts.safety_prompt import build_safety_prompt
from safenav_vlm.utils.image_preprocessing import resize_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_response(response: str) -> tuple[str, str]:
    """Extract the safety decision and reason from the model response."""

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
        description="Run the SafeNav-VLM direct baseline on multiple images."
    )

    parser.add_argument(
        "--input-dir",
        default="data/pilot/images",
    )

    parser.add_argument(
        "--processed-dir",
        default="data/processed/baseline",
    )

    parser.add_argument(
        "--output",
        default="results/predictions/direct_baseline.csv",
    )

    parser.add_argument(
        "--question",
        default="Can I safely walk forward?",
    )

    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    processed_dir = Path(args.processed_dir)
    output_path = Path(args.output)

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.name.startswith("test_image")
    )

    if not image_paths:
        raise RuntimeError(f"No test images found in {input_dir}")

    print(f"Found {len(image_paths)} images.")
    print("Loading VLM once...")

    vlm = QwenVLM()
    prompt = build_safety_prompt(args.question)

    results = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}] {image_path.name}")

        with Image.open(image_path) as image:
            original_width, original_height = image.size

        processed_path = (
            processed_dir
            / f"{image_path.stem}_1024.jpg"
        )

        resize_image(
            image_path,
            processed_path,
            max_dimension=args.max_dimension,
        )

        with Image.open(processed_path) as image:
            processed_width, processed_height = image.size

        start = time.perf_counter()

        response = vlm.generate_response(
            image_path=str(processed_path),
            prompt=prompt,
        )

        inference_time = time.perf_counter() - start

        decision, reason = parse_response(response)

        print(f"Decision: {decision}")
        print(f"Reason: {reason}")
        print(f"Inference: {inference_time:.2f}s")

        results.append(
            {
                "image": image_path.name,
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": processed_width,
                "processed_height": processed_height,
                "question": args.question,
                "decision": decision,
                "reason": reason,
                "inference_seconds": round(inference_time, 3),
                "model": vlm.model_name,
            }
        )

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n--------------------------------")
    print("Batch complete")
    print(f"Results saved to: {output_path}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
