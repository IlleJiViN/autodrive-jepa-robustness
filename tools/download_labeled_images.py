"""
Downloads the exact 2,304 labeled BDD100K images from the Hugging Face mirror
and prepares local manifest files with correct Windows paths.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
from PIL import Image

REPO = "karthik789338/BDD100K-Images"
REVISION = "8f6ba5d675567ca3a98c50947bdd83524c4749f8"
ROOT = Path(__file__).resolve().parents[1]
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"


def request(url: str):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "autodrive-jepa-research/0.1"})
            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read()
                return content
        except Exception as error:
            if attempt == 5:
                raise RuntimeError(f"Download failed after retries: {url}: {error}")
            time.sleep(min(2**attempt, 16))


def download_one(row: dict, split: str) -> bool:
    source_path = row.get("source_path")
    if not source_path:
        fname = Path(row["image_path"]).name
        source_path = f"images/100k/{split}/{fname}"

    filename = Path(source_path).name
    target_path = ROOT / "data/bdd100k/images/100k" / split / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size > 0:
        return False

    url = BASE + urllib.parse.quote(source_path, safe="/")
    content = request(url)
    digest = hashlib.sha256(content).hexdigest()
    expected = row.get("image_sha256")
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch for {source_path}")

    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(target_path)
    return True


def process_manifest(manifest_path: Path, split: str, max_workers: int = 8, limit: int = None):
    print(f"\n[DataSync] Loading {manifest_path.name}...")
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    if limit:
        records = records[:limit]

    print(f"[DataSync] Checking {len(records)} images for split: {split}...")
    started = time.time()
    downloaded = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_one, row, split): row for row in records}
        for done, f in enumerate(concurrent.futures.as_completed(futures), 1):
            if f.result():
                downloaded += 1
            if done % 200 == 0 or done == len(records):
                elapsed = time.time() - started
                print(f"  -> Processed {done}/{len(records)} | Downloaded new: {downloaded} | Elapsed: {elapsed:.1f}s")

    # Generate local path manifest
    local_manifest = ROOT / f"data/manifests/bdd100k_{split}_local.jsonl"
    local_records = []
    for row in records:
        source_path = row.get("source_path")
        fname = Path(source_path).name if source_path else Path(row["image_path"]).name
        local_path = str(ROOT / "data/bdd100k/images/100k" / split / fname)
        new_row = dict(row)
        new_row["image_path"] = local_path
        local_records.append(new_row)

    local_manifest.write_text("\n".join(json.dumps(r) for r in local_records) + "\n", encoding="utf-8")
    print(f"[DataSync] Local manifest saved: {local_manifest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for quick testing)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    train_mf = ROOT / "data/manifests/bdd100k_train_labeled.jsonl"
    val_mf = ROOT / "data/manifests/bdd100k_val_labeled.jsonl"

    if not train_mf.exists() or not val_mf.exists():
        raise FileNotFoundError("Labeled manifests not found in data/manifests/")

    # Val first (256 images)
    process_manifest(val_mf, "val", max_workers=args.workers, limit=args.limit)
    # Train next (2,048 images)
    process_manifest(train_mf, "train", max_workers=args.workers, limit=args.limit)
    print("\n[DataSync] All labeled images and local manifests are ready!")


if __name__ == "__main__":
    main()
