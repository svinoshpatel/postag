# Dictionary reference: VESUM / dict_uk

[source.json](source.json) pins the release, URL, format, SHA-256, attribution,
and license. `dict_corp_vis.txt.bz2` is the downloaded expanded-form dictionary;
it is ignored by Git and can be obtained from the repository root with:

```bash
python scripts/fetch_dictionary.py
```

The downloader verifies the archive before publishing it. Experiments using the
default dictionary verify it again and stream its analyses without extracting a
large text file. No Java build is required.

The dictionary data are by Andriy Rysin, Vasyl Starko, and the BrUK team under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The pinned upstream release is recorded in `source.json`.

Evaluation unions all coarse POS analyses for a normalized form. Ambiguous forms
are counted but excluded from primary scoring; dictionary lookup is not a
contextual POS disambiguator. See the
[mapping and evaluation guide](../../docs/EXPERIMENTS.md#dictionary-source-and-scoring-policy).
