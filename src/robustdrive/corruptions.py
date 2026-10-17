from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

FAMILIES = ("fog", "rain", "motion_blur", "low_light")


def corrupt(image: Image.Image, family: str, severity: int, seed: int) -> Image.Image:
    """Deterministic synthetic corruptions; severity scales are heuristic defaults."""
    if family == "clean":
        return image.copy()
    if family not in FAMILIES or severity not in range(1, 6):
        raise ValueError(f"Unsupported corruption: {family}/{severity}")
    rng = np.random.default_rng(seed)
    values = np.asarray(image, dtype=np.float32) / 255.0
    height, width = values.shape[:2]
    strength = severity / 5.0
    if family == "fog":
        lowres = Image.fromarray(rng.integers(40, 255, (12, 20), dtype=np.uint8))
        field = np.asarray(lowres.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        opacity = np.clip(strength * (0.25 + 0.6 * field), 0, 0.85)[..., None]
        values = values * (1 - opacity) + 0.95 * opacity
    elif family == "rain":
        layer = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(layer)
        for _ in range(int(width * height * (0.0005 + 0.003 * strength))):
            x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
            length = int(4 + strength * 18)
            draw.line((x, y, x - max(1, length // 3), y + length), fill=int(rng.integers(90, 230)), width=1)
        streaks = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.4 + strength * 0.4)), dtype=np.float32) / 255.0
        alpha = (streaks * (0.25 + strength * 0.5))[..., None]
        values = values * (1 - alpha) + alpha * 0.85
    elif family == "motion_blur":
        length = 3 + 2 * severity
        radius = length // 2
        padded = np.pad(values, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        values = sum(padded[:, offset:offset + width] for offset in range(length)) / length
    elif family == "low_light":
        values = values ** (1.0 + 1.8 * strength) * (1 - 0.65 * strength)
        values += rng.normal(0, 0.002 + strength * 0.018, values.shape).astype(np.float32)
    return Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255))
