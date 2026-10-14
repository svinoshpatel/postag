# Utility scripts

Standalone commands for acquiring project inputs. Run from the repository root:

```bash
python scripts/fetch_dictionary.py
```

[fetch_dictionary.py](fetch_dictionary.py) reads the pinned release URL and hash
from [data/dict_uk/source.json](../data/dict_uk/source.json), downloads the expanded
bzip2 dictionary, verifies SHA-256, then publishes the completed file. A matching
existing file is reused offline; an incorrect existing file is not overwritten.

The downloader uses only Python's standard library. Processing commands are in
the [postag package](../postag/README.md), invoked with `python -m postag...`.
