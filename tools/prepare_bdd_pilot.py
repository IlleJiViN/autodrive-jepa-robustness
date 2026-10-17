"""Download a pinned, image-only BDD100K pilot subset from a public mirror."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

REPO = "karthik789338/BDD100K-Images"
REVISION = "8f6ba5d675567ca3a98c50947bdd83524c4749f8"
ROOT = Path(__file__).resolve().parents[1]
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"


def request(url):
    for attempt in range(6):
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "autodrive-jepa-research/0.1"}), timeout=60)
            content, headers = response.read(), dict(response.headers)
            response.close()
            return content, headers
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 5:
                raise RuntimeError(f"Download failed after retries: {url.split('?')[0]}: {error}") from error
            time.sleep(min(2**attempt, 16))


def tree(path, recursive=False):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{path}?recursive={'true' if recursive else 'false'}&limit=1000"
    entries = []
    while url:
        raw, headers = request(url)
        entries.extend(json.loads(raw))
        links = headers.get("Link", headers.get("link", ""))
        url = next((part.split("<", 1)[1].split(">", 1)[0] for part in links.split(",") if 'rel="next"' in part), None)
    return entries


def candidates(split):
    parent = f"images/100k/{split}"
    top = tree(parent)
    files = [row for row in top if row["type"] == "file" and row["path"].lower().endswith(".jpg")]
    folders = [row["path"] for row in top if row["type"] == "directory"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(lambda path: tree(path, recursive=True), folders):
            files.extend(row for row in rows if row["type"] == "file" and row["path"].lower().endswith(".jpg"))
    print(json.dumps({"event": "source_index", "split": split, "images": len(files), "shards": len(folders)}), flush=True)
    return sorted(files, key=lambda row: row["path"])


def download_image(task):
    split, entry = task
    path = ROOT / "data/bdd100k/images/100k" / split / Path(entry["path"]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + urllib.parse.quote(entry["path"], safe="/")
    if path.is_file():
        content = path.read_bytes()
    else:
        content, _ = request(url)
    digest = hashlib.sha256(content).hexdigest()
    expected = entry.get("lfs", {}).get("oid")
    if expected:
        if digest != expected:
            raise ValueError(f"Mirror SHA-256 mismatch: {entry['path']}")
    elif hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() != entry["oid"]:
        raise ValueError(f"Mirror Git blob mismatch: {entry['path']}")
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    return {"image_path": str(path), "scene_id": path.stem, "split": split,
            "boxes": [], "labels": [], "annotation_status": "unavailable_image_only",
            "width": width, "height": height, "image_sha256": digest,
            "source_repo": REPO, "source_revision": REVISION, "source_path": entry["path"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", type=int, default=2048)
    parser.add_argument("--val-images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    counts = {"train": args.train_images, "val": args.val_images}
    if min(counts.values()) < 2:
        raise ValueError("Each split needs at least two images")
    tasks, available = [], {}
    for split, count in counts.items():
        rows = candidates(split)
        available[split] = len(rows)
        if count > len(rows):
            raise ValueError(f"Requested more than available {split} images")
        selected = random.Random(args.seed + (0 if split == "train" else 1)).sample(rows, count)
        tasks.extend((split, row) for row in selected)
    records = {split: [] for split in counts}
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_image, task) for task in tasks]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            records[record["split"]].append(record)
            if done % 100 == 0 or done == len(tasks):
                print(json.dumps({"event": "download", "done": done, "total": len(tasks),
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    training_groups = {row["scene_id"] for row in records["train"]}
    validation_groups = {row["scene_id"] for row in records["val"]}
    training_hashes = {row["image_sha256"] for row in records["train"]}
    validation_hashes = {row["image_sha256"] for row in records["val"]}
    if training_groups & validation_groups or training_hashes & validation_hashes:
        raise ValueError("Train/val scene IDs or image bytes overlap")
    destination = ROOT / "data/manifests"
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        text = "".join(json.dumps(row) + "\n" for row in sorted(rows, key=lambda row: row["scene_id"]))
        output = destination / f"bdd100k_{split}.jsonl"
        if output.exists() and output.read_text() != text:
            raise FileExistsError(f"Existing manifest differs: {output}")
        output.write_text(text, encoding="utf-8")
    provenance = ROOT / "data/bdd100k"
    for name in ("README.md", "LICENSE_NOTICE.md"):
        raw, _ = request(BASE + name)
        (provenance / ("SOURCE_" + name)).write_bytes(raw)
    summary = {"status": "pilot_images_ready", "source": "public_mirror_not_official_download_server",
               "repo": REPO, "revision": REVISION, "source_split_paths": "images/100k/train and images/100k/val",
               "available_images": available, "downloaded_images": counts, "seed": args.seed,
               "all_downloads_verified_against_mirror_hashes": True, "cross_split_content_overlap": 0,
               "detection_annotations_available": False, "sampling": "seeded_random_subset_of_each_source_split",
               "official_archive_byte_equality_verified": False, "preparation_seconds": time.perf_counter() - started}
    (provenance / "pilot_provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
