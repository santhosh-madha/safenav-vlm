"""Image preprocessing utilities for SafeNav-VLM."""

from pathlib import Path

from PIL import Image, ImageOps


def resize_image(
    input_path: str | Path,
    output_path: str | Path,
    max_dimension: int,
    quality: int = 95,
) -> Path:
    """Resize an image while preserving its aspect ratio.

    The longest side is limited to max_dimension pixels.
    Images smaller than the requested size are not enlarged.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        # Respect camera EXIF orientation before resizing.
        image = ImageOps.exif_transpose(image)

        original_width, original_height = image.size

        scale = min(
            1.0,
            max_dimension / max(original_width, original_height),
        )

        new_width = round(original_width * scale)
        new_height = round(original_height * scale)

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        # JPEG cannot store RGBA.
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            resized = resized.convert("RGB")

        resized.save(output_path, quality=quality)

    return output_path
