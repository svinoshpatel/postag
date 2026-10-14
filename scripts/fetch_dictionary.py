"""Download the pinned dict_uk release and verify its recorded SHA-256."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen


DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "dict_uk"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source = json.loads((DIRECTORY / "source.json").read_text(encoding="utf-8"))
    destination = DIRECTORY / source["filename"]
    if destination.exists():
        if sha256(destination) != source["sha256"]:
            raise ValueError(f"Existing dictionary checksum is incorrect: {destination}. "
                             "Move it aside before downloading again.")
        print(f"Verified existing dictionary: {destination}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=DIRECTORY, prefix=".download-", delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(source["url"], timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
        if sha256(temporary) != source["sha256"]:
            raise ValueError("Downloaded dictionary checksum differs from the pinned release")
        # Publish without overwriting a file that appeared during the download.
        os.link(temporary, destination)
        print(f"Downloaded and verified {source['release']}: {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
