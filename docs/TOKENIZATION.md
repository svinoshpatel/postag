# Ukrainian tokenization foundations

The installable core is `crf`, under `src/crf/`. Run commands from the repository
root after `uv sync --locked`. `.python-version` selects CPython 3.14.6;
`uv.lock` records the exact dependency environment. No trained language model
or spaCy installation is required.

## Preview a paragraph

```bash
uv run python scripts/preview_tokenization.py --text "Це проф. Петренко. Він у Києві."
```

The argparse entry point delegates to `crf.data.tokenization.tokenize_paragraph`
and prints UTF-8-capable JSON to stdout. Each sentence contains a `sentence`
span and `tokens` spans with `text`, `start`, and `end`. Offsets are zero-based,
end-exclusive Unicode code-point positions in the supplied paragraph. The
Python API returns frozen dataclasses with tuple collections.

```python
from crf.data.tokenization import tokenize_paragraph

for item in tokenize_paragraph("Її сім’я живе в Києві."):
    print(item.sentence.text)
    print([token.text for token in item.tokens])
```

The backend uses `tokenize-uk==2.0.0` non-legacy sentence and word span APIs,
with `language_code="uk_two"`. Original text is recovered by slicing the input
at returned offsets because upstream token strings may normalize apostrophes.
Single line breaks do not independently terminate sentences. Empty/whitespace
results are omitted. Abbreviation boundaries follow the upstream Ukrainian rules.

## Implementation boundary

This is a low-level backend and a text preview, not the production file-processing
adapter or prediction CLI. It holds the supplied paragraph in memory. Callers
must not pass whole books and treat this as bounded corpus preprocessing.

Normalization with original-source alignment, paragraph buffering, size limits,
UD-specific punctuation/hyphen/abbreviation adaptation, formatting-character
reporting, and corpus compatibility checks remain in milestones 3–4 of
[PLAN.md](../PLAN.md). In particular, upstream span APIs can omit standalone
format characters; the future adapter must preserve/report these according to
[the architecture](ARCH.md). Gold segmentation is never supplied by this backend.

The previous `python -m postag.tokenization` interface and its whole-file,
recursive, overwrite-capable processing have been retired. The new preview
accepts only `--text` and cannot write generated dataset files. Existing
exploratory outputs remain untouched and are not compatible pipeline artifacts.

## Verification

```bash
uv run pytest tests/test_tokenization.py tests/test_environment.py -v
```

Tests use tiny strings and temporary files. They verify abbreviation-aware
sentence boundaries, original apostrophe slices, ordered code-point offsets,
empty inputs, immutability, installed-package imports, and CLI JSON. The
integration-marked environment test also checks CRFsuite fitting, marginals,
and serialization without network access or production data.

Upstream API reference: [tokenize-uk](https://github.com/lang-uk/tokenize-uk/blob/master/docs/api.md).
