"""Download and read the open PTB Diagnostic ECG Database."""

from __future__ import annotations

import csv
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wfdb

from .labels import clinical_fields, label_from_comments


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "mi-vcg-tucker/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("latin1")


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mi-vcg-tucker/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            tmp.replace(destination)
            return
        except Exception as error:  # pragma: no cover - network dependent
            last_error = error
            tmp.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}") from last_error


def prepare_database(base_url: str, raw_dir: Path) -> list[dict[str, str]]:
    """Download headers and only the Frank XYZ signal files used by the paper."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = fetch_text(f"{base_url.rstrip('/')}/RECORDS").split()
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = {
            pool.submit(
                download_file,
                f"{base_url.rstrip('/')}/{record}.hea",
                raw_dir / Path(record).with_suffix(".hea"),
            ): record
            for record in records
        }
        for index, future in enumerate(as_completed(jobs), 1):
            future.result()
            if index % 50 == 0:
                print(f"Downloaded {index}/{len(records)} headers", flush=True)
    manifest: list[dict[str, str]] = []
    for record in records:
        relative = Path(record)
        header_path = raw_dir / relative.with_suffix(".hea")
        header_text = header_path.read_text(encoding="latin1")
        comments = [line for line in header_text.splitlines() if line.startswith("#")]
        label = label_from_comments(comments)
        if label is None:
            continue
        fields = clinical_fields(comments)
        manifest.append({
            "record": record,
            "patient_id": relative.parent.name,
            "label": label,
            "label_source": (
                "acute" if label_from_comments([f"# Reason for admission: Myocardial infarction", f"# Acute infarction (localization): {fields.get('acute infarction (localization)', '')}"]) else "former"
            ) if label != "HC" else "healthy",
            "acute_location": fields.get("acute infarction (localization)", ""),
            "former_location": fields.get("former infarction (localization)", ""),
        })
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {
            pool.submit(
                download_file,
                f"{base_url.rstrip('/')}/{item['record']}.xyz",
                raw_dir / Path(item["record"]).with_suffix(".xyz"),
            ): item["record"]
            for item in manifest
        }
        for index, future in enumerate(as_completed(jobs), 1):
            future.result()
            if index % 25 == 0:
                print(f"Downloaded {index}/{len(manifest)} VCG records", flush=True)
    manifest_path = raw_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_vcg(raw_dir: Path, record: str, leads: list[str]) -> tuple[object, int]:
    rec = wfdb.rdrecord(str(raw_dir / record), channel_names=leads)
    return rec.p_signal, int(rec.fs)
