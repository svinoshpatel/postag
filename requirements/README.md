# Dependency specifications

Install from the repository root inside the project virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements/tokenization.txt -r requirements/experiment.txt
```

| File | Used by |
| --- | --- |
| [tokenization.txt](tokenization.txt) | Ukrainian tokenizer, spaCy integration, and tokenizer tests. |
| [experiment.txt](experiment.txt) | Sparse encoding, SVD, scikit-learn HDBSCAN, evaluation, and experiment tests. |

The pins were tested together on Python 3.9.6. Install only the experiment set
when working with existing tokenized data; install both to run the full test suite.
HDBSCAN comes from scikit-learn; the separate `hdbscan` package is not required.
The dictionary downloader uses the standard library. Dictionary files and corpus
data are not installed by pip; see [data/](../data/README.md).
