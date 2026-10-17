import os
import sys
import time
import math
import zipfile
import threading
import urllib.request
from pathlib import Path

TARGET_URL = "https://huggingface.co/datasets/Xing1210/bdd100k/resolve/main/bdd100k_images_100k.zip"
EXPECTED_BYTES = 5669071832
NUM_WORKERS = 8
BUFFER_SIZE = 2 * 1024 * 1024  # 2MB buffer

def get_final_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.url

def download_worker(worker_id, cdn_url, start_byte, end_byte, fd, progress_dict, lock):
    current = start_byte
    retries = 5
    while current <= end_byte and retries > 0:
        req = urllib.request.Request(
            cdn_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={current}-{end_byte}"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Worker {worker_id} unexpected HTTP status {resp.status}")
                while current <= end_byte:
                    chunk_len = min(BUFFER_SIZE, end_byte - current + 1)
                    buf = resp.read(chunk_len)
                    if not buf:
                        break
                    os.pwrite(fd, buf, current)
                    current += len(buf)
                    with lock:
                        progress_dict[worker_id] = current - start_byte
        except Exception as e:
            retries -= 1
            print(f"[Worker {worker_id}] Error: {e}, retrying ({retries} left)...", flush=True)
            time.sleep(2)

    if current <= end_byte:
        raise RuntimeError(f"Worker {worker_id} failed to complete chunk: {current}/{end_byte}")

def main():
    dest_path = Path("/home/coder/autodrive-jepa-robustness/data/bdd100k/bdd100k_images_100k.zip")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists() and dest_path.stat().st_size == EXPECTED_BYTES:
        print(f"Archive already exists and has correct size {EXPECTED_BYTES} bytes. Verifying...", flush=True)
        try:
            with zipfile.ZipFile(dest_path) as zf:
                print(f"Zip verified. Top 5 entries: {zf.namelist()[:5]}", flush=True)
                return
        except Exception as e:
            print(f"Existing file corrupted ({e}), redownloading...", flush=True)

    part_path = dest_path.with_suffix(".zip.part")
    print(f"Resolving download URL...", flush=True)
    cdn_url = get_final_url(TARGET_URL)
    print(f"Resolved CDN endpoint: {cdn_url[:80]}...", flush=True)

    print(f"Pre-allocating {EXPECTED_BYTES} bytes at {part_path}...", flush=True)
    fd = os.open(str(part_path), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, EXPECTED_BYTES)

    chunk_size = math.ceil(EXPECTED_BYTES / NUM_WORKERS)
    threads = []
    progress_dict = {i: 0 for i in range(NUM_WORKERS)}
    lock = threading.Lock()

    start_time = time.time()
    for i in range(NUM_WORKERS):
        start_byte = i * chunk_size
        end_byte = min((i + 1) * chunk_size - 1, EXPECTED_BYTES - 1)
        t = threading.Thread(
            target=download_worker,
            args=(i, cdn_url, start_byte, end_byte, fd, progress_dict, lock),
            daemon=True
        )
        t.start()
        threads.append(t)

    # Monitor progress
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        with lock:
            downloaded = sum(progress_dict.values())
        elapsed = time.time() - start_time
        speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
        pct = (downloaded / EXPECTED_BYTES) * 100
        print(f"[{elapsed:5.1f}s] {downloaded / 1e9:5.2f} / {EXPECTED_BYTES / 1e9:5.2f} GB ({pct:5.1f}%) @ {speed_mb:6.1f} MB/s", flush=True)

    for t in threads:
        t.join()

    os.close(fd)
    final_size = part_path.stat().st_size
    print(f"Download completed in {time.time() - start_time:.1f}s. Final size: {final_size} bytes (expected {EXPECTED_BYTES})", flush=True)
    if final_size != EXPECTED_BYTES:
        raise ValueError(f"Size mismatch: {final_size} != {EXPECTED_BYTES}")

    print("Verifying zip archive header and index...", flush=True)
    with zipfile.ZipFile(part_path) as zf:
        sample = zf.namelist()[:5]
        print(f"Sample namelist: {sample}", flush=True)
        print(f"Total files in zip: {len(zf.namelist())}", flush=True)

    part_path.replace(dest_path)
    print(f"Successfully saved {dest_path} ({dest_path.stat().st_size} bytes)!", flush=True)

if __name__ == "__main__":
    main()