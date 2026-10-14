# Experiment outputs

Each named child directory is one self-contained run. Start with its README and
`report.md`; deeper directories document feature variants, SVD vectors, and
HDBSCAN assignments. New completed runs generate those READMEs automatically.

Existing runs:

| Directory | Purpose |
| --- | --- |
| [baseline/](baseline/README.md) | Unscaled comparison on 6,000 sampled occurrences. |
| [scaled/](scaled/README.md) | Same sample with numeric standard-deviation scaling. |
| [pilot/](pilot/README.md) | Preliminary run retained for provenance. |

`artifact-audit.json` records validation performed before the directory move.
See [the measured results](../docs/RESULTS.md) for interpretation.

Create a new run from the repository root:

```bash
python -m postag.experiment --output runs/my-experiment --sample-size 6000
```

Existing output directories are not overwritten. Binary artifacts and per-run
outputs are ignored by Git; this directory's README is tracked. Keep each run's
rows, matrices, labels, and manifest together because they share row order.

Load trusted encoders and reducers with `postag.artifacts.load_transform`. This
also supports existing encoders that reference the old `vectors` module.
Historical manifests retain their original absolute input paths and code hashes;
the corpus now resides at `data/tokenized/`, and source modules are in `postag/`.
No original experiment artifacts were rewritten during the move.
