# Python package

This directory contains the processing and experiment code. Run commands from
the repository root so Python can import the `postag` package.

| Module | Responsibility |
| --- | --- |
| [tokenization.py](tokenization.py) | Split text into sentences/tokens while preserving case and punctuation. |
| [features.py](features.py) | Shared token/neighbor features, TF helpers, and feature configuration. Formerly `main.py`. |
| [corpus.py](corpus.py) | Validate corpus JSON, count TF, sample occurrences, retain source positions. |
| [vectors.py](vectors.py) | Select feature variants, encode sparse vectors, apply SVD and optional scaling. |
| [evaluation.py](evaluation.py) | Dictionary POS lookup, ambiguity/coverage, partition metrics, cluster examples. |
| [experiment.py](experiment.py) | CLI, parameter sweeps, persisted matrices and reports. |
| [artifacts.py](artifacts.py) | Load current/legacy transforms and create run-directory READMEs. |

```bash
python -m postag.tokenization --help
python -m postag.experiment --help
```

Use ordinary package imports from code:

```python
from postag.features import build_word_frequencies, sent2features
from postag.artifacts import load_transform

sentences = [["Київ", "гарний", "."]]
tf = build_word_frequencies(sentences)
features = sent2features(sentences[0], tf)
encoder = load_transform("runs/baseline/context_tf/encoder.joblib")
X = encoder.transform(features)
```

`load_transform` is for trusted local joblib files. It supports encoders saved
before the package move without altering their bytes. New artifacts directly
reference `postag.vectors.FeatureEncoder`.

Default data/output paths are anchored to the repository root. Explicit CLI paths
are relative to the working directory. Source fingerprints in new manifests use
package-relative paths such as `postag/features.py`.

See the [experiment guide](../docs/EXPERIMENTS.md),
[tokenization guide](../docs/TOKENIZATION.md), and [tests](../tests/README.md).
