from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from robustdrive.common import stable_seed
from robustdrive.corruptions import corrupt
from robustdrive.foundation_v2 import CLASS_NAMES, FoundationViTDetectorV2, decode_v2

ROOT = Path(__file__).resolve().parent.parent

# Distinct high-contrast colors for key driving classes (RGB)
CLASS_COLORS = {
    0: (0, 255, 128),    # car: bright mint green
    1: (255, 50, 50),    # pedestrian: bright red/coral
    2: (0, 200, 255),    # rider: cyan
    3: (255, 180, 0),    # truck: orange-gold
    4: (255, 220, 0),    # bus: yellow
    5: (200, 100, 255),  # train: purple
    6: (255, 100, 180),  # motorcycle: pink
    7: (50, 255, 255),   # bicycle: electric cyan
    8: (255, 255, 255),  # traffic light: white
    9: (255, 255, 0),    # traffic sign: yellow
}
DEFAULT_COLOR = (0, 255, 255)


def draw_predictions_on_image(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    score_thresh: float = 0.35,
    title: str = "",
) -> Image.Image:
    """Draws bounding boxes and labels on an image with a dark header banner."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    # Try to load a nice truetype font, fallback to default bitmap font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            title_font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
            title_font = font

    # Draw boxes
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < score_thresh:
            continue
        cls_id = int(labels[i])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        box = boxes[i]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        # Bounding box (width = 3)
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        # Label tag with background
        tag = f"{cls_name} {score:.2f}"
        bbox = draw.textbbox((x1, y1), tag, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tag_y1 = max(0, y1 - text_h - 4)
        draw.rectangle([x1, tag_y1, x1 + text_w + 6, tag_y1 + text_h + 4], fill=color)
        draw.text((x1 + 3, tag_y1 + 1), tag, fill=(0, 0, 0), font=font)

    # Top banner with title
    if title:
        banner_h = 32
        banner = Image.new("RGBA", (w, banner_h), (10, 15, 25, 220))
        canvas.paste(banner, (0, 0), banner)
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 5), title, fill=(255, 255, 255), font=title_font)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate Visual Detection Comparison Gallery")
    parser.add_argument("--checkpoint", type=str, default="runs/rtx5090-v2-full70k-12ep/best_checkpoint.pt")
    parser.add_argument("--val-manifest", type=str, default="data/manifests/bdd100k_val_local.jsonl")
    parser.add_argument("--output-dir", type=str, default="docs/assets")
    parser.add_argument("--score-thresh", type=float, default=0.35)
    parser.add_argument("--scene-indices", type=int, nargs="+", default=[1, 4, 12])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AutoDrive-JEPA: Generating Visual Qualitative Comparison Gallery")
    print(f"Device: {args.device} | Score Thresh: {args.score_thresh}")
    print("=" * 70)

    # Load model
    print(f"Loading detector from {args.checkpoint}...")
    model = FoundationViTDetectorV2(model_name="vit_small_patch16_384", pretrained=False).to(args.device)
    sd = torch.load(args.checkpoint, map_location=args.device)
    model_state = sd.get("model_state", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(model_state)
    model.eval()

    val_records = [json.loads(l) for l in Path(args.val_manifest).read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    print(f"Loaded {len(val_records):,} validation scenes.")

    target_size = (384, 640)  # (H, W)

    conditions = [
        ("Clean (Daytime / Baseline)", None, 0),
        ("Adverse Weather: Fog (Severity 2)", "fog", 2),
        ("Adverse Weather: Rain (Severity 2)", "rain", 2),
        ("Adverse Weather: Low-Light (Severity 2)", "low_light", 2),
    ]

    for s_idx in args.scene_indices:
        if s_idx >= len(val_records):
            continue
        rec = val_records[s_idx]
        scene_id = rec.get("scene_id", f"scene_{s_idx}")
        print(f"\nProcessing Scene {s_idx} ({scene_id})...")

        with Image.open(rec["image_path"]) as img:
            orig_img = img.convert("RGB")
        orig_w, orig_h = orig_img.size

        rendered_panels = []

        for title, fam, sev in conditions:
            if fam is None:
                panel_img = orig_img.copy()
            else:
                c_seed = stable_seed(42, str(rec["image_path"]), fam, sev)
                panel_img = corrupt(orig_img, fam, severity=sev, seed=c_seed)

            # Resize to model input size for inference
            resized = panel_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            inp = to_tensor(resized).unsqueeze(0).to(args.device)

            with torch.no_grad():
                with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    preds = model(inp)
                decoded = decode_v2(preds, image_size=target_size)[0]

            # Scale detected boxes back to original image dimensions for high-res rendering
            sx = orig_w / float(target_size[1])
            sy = orig_h / float(target_size[0])
            boxes = decoded["boxes"] if isinstance(decoded["boxes"], np.ndarray) else decoded["boxes"].cpu().numpy()
            if len(boxes) > 0:
                scaled_boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
            else:
                scaled_boxes = np.empty((0, 4), dtype=np.float32)

            scores = decoded["scores"] if isinstance(decoded["scores"], np.ndarray) else decoded["scores"].cpu().numpy()
            labels = decoded["labels"] if isinstance(decoded["labels"], np.ndarray) else decoded["labels"].cpu().numpy()

            num_det = int(np.sum(scores >= args.score_thresh))
            sub_title = f"{title} — {num_det} objects detected"
            panel = draw_predictions_on_image(
                panel_img,
                scaled_boxes,
                scores,
                labels,
                score_thresh=args.score_thresh,
                title=sub_title,
            )
            rendered_panels.append(panel)

        # Stitch into a 2x2 high-res grid
        pw, ph = rendered_panels[0].size
        grid = Image.new("RGB", (pw * 2, ph * 2))
        grid.paste(rendered_panels[0], (0, 0))
        grid.paste(rendered_panels[1], (pw, 0))
        grid.paste(rendered_panels[2], (0, ph))
        grid.paste(rendered_panels[3], (pw, ph))

        grid_path = out_dir / f"weather_detection_comparison_scene_{s_idx}.png"
        grid.save(grid_path, quality=95)
        print(f"  Saved 2x2 comparison grid: {grid_path} ({grid.size[0]}x{grid.size[1]})")

        # Also save the primary showcase image as the main asset
        if s_idx == args.scene_indices[0]:
            main_path = out_dir / "weather_detection_comparison.png"
            grid.save(main_path, quality=95)
            print(f"  >>> Saved primary showcase image: {main_path} <<<")

    print("\nVisual qualitative generation complete!")


if __name__ == "__main__":
    main()
