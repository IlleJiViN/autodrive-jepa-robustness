import os
import sys
import time
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ARCHIVE = Path("/home/coder/autodrive-jepa-robustness/data/bdd100k/bdd100k_images_100k.zip")
DEST_DIR = Path("/home/coder/autodrive-jepa-robustness/data/bdd100k/images")

def extract_batch(items):
    # items: list of (member_name, expected_size)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(ARCHIVE) as zf:
        for member_name, expected_size in items:
            out_path = DEST_DIR / member_name
            if out_path.exists() and out_path.stat().st_size == expected_size:
                skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member_name) as src, open(out_path, "wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            extracted += 1
    return extracted, skipped

def main():
    print(f"Scanning archive {ARCHIVE}...", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(ARCHIVE) as zf:
        train_members = []
        val_members = []
        for info in zf.infolist():
            if info.filename.startswith("100k/train/") and info.filename.endswith(".jpg"):
                train_members.append((info.filename, info.file_size))
            elif info.filename.startswith("100k/val/") and info.filename.endswith(".jpg"):
                val_members.append((info.filename, info.file_size))

    print(f"Found {len(train_members)} train images and {len(val_members)} val images in {time.time()-t0:.2f}s", flush=True)
    all_members = train_members + val_members
    print(f"Total images to extract/verify: {len(all_members)}", flush=True)

    batch_size = 500
    batches = [all_members[i:i+batch_size] for i in range(0, len(all_members), batch_size)]
    
    t1 = time.time()
    total_extracted = 0
    total_skipped = 0
    with ProcessPoolExecutor(max_workers=24) as pool:
        for extracted, skipped in pool.map(extract_batch, batches):
            total_extracted += extracted
            total_skipped += skipped
            done = total_extracted + total_skipped
            if done % 10000 == 0 or done == len(all_members):
                elapsed = time.time() - t1
                rate = done / max(elapsed, 0.001)
                print(f"Progress: {done}/{len(all_members)} ({done/len(all_members)*100:.1f}%) | "
                      f"Extracted: {total_extracted}, Skipped: {total_skipped} | Rate: {rate:.1f} img/s", flush=True)

    print(f"Extraction complete in {time.time()-t1:.2f}s!", flush=True)
    
    train_count = len(list((DEST_DIR / "100k/train").glob("*.jpg")))
    val_count = len(list((DEST_DIR / "100k/val").glob("*.jpg")))
    print(f"Final disk counts -> Train: {train_count}, Val: {val_count}", flush=True)
    if train_count != 70000 or val_count != 10000:
        raise ValueError(f"Unexpected image count! Expected 70000 train and 10000 val, got {train_count} and {val_count}")
    print("SUCCESS: Image extraction verification passed!", flush=True)

if __name__ == "__main__":
    main()
