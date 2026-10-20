#!/usr/bin/env python3
"""Show verified bytes and conservative aria2 partial progress without network I/O."""
import json
from pathlib import Path
import struct

project = Path(__file__).resolve().parents[1]
manifest = json.loads((project / "data/training/manifest.json").read_text())
partial = 0
for path in (project / "logs/download_training_data/partials").iterdir():
    if path.name.endswith((".aria2", ".log")):
        continue
    control = path.with_name(path.name + ".aria2")
    try:
        if control.exists():
            data = control.read_bytes()
            offset = 10 + struct.unpack_from(">I", data, 6)[0]
            piece_size, total, _, count = struct.unpack_from(">IQQI", data, offset)
            bits = data[offset + 24:offset + 24 + count]
            partial += min(total, sum(byte.bit_count() for byte in bits) * piece_size)
        else:
            partial += path.stat().st_size
    except (FileNotFoundError, struct.error):
        pass  # A completed part can disappear while the downloader moves it.
verified = sum(item["verified_bytes"] for item in manifest["datasets"])
expected = sum(item["expected_bytes"] for item in manifest["datasets"])
print(json.dumps({"updated_at": manifest["updated_at"],
                  "verified_GiB": round(verified / 2**30, 3),
                  "partial_GiB": round(partial / 2**30, 3),
                  "total_GiB": round(expected / 2**30, 3),
                  "datasets": [{"name": item["directory"], "status": item["status"],
                                "files": f"{item['verified_files']}/{item['expected_files']}",
                                "verified_GiB": round(item["verified_bytes"] / 2**30, 3)}
                               for item in manifest["datasets"]]}, ensure_ascii=False, indent=2))
