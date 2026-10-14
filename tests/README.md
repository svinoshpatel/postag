# Tests

Run from the repository root after installing both dependency files:

```bash
python -B -m unittest discover -s tests -v
```

| File | Coverage |
| --- | --- |
| [test_tokenization.py](test_tokenization.py) | Sentence boundaries, text preservation, encodings, directory discovery, output protection. |
| [test_features.py](test_features.py) | Affixes, case, TF, neighbors, boundaries, and annotation independence. |
| [test_experiment.py](test_experiment.py) | Sampling/alignment, vector persistence, dictionary parsing, metrics/noise, end-to-end artifacts, and legacy loading. |

Tests import the `postag` package and use temporary directories. Synthetic test
dictionaries do not require network access or the full reference download. Test
downloads use local fixture files. Existing corpus and run artifacts are not
modified. The real-corpus experiment results are documented separately in
[RESULTS.md](../docs/RESULTS.md).
