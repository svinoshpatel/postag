# Raw Ukrainian corpus

This is the original 200-file corpus, previously named `ukrainian_200texts/` at
the repository root. File contents and basenames are unchanged. The original
Google Drive provenance was not verified during implementation.

The tokenizer reads `.txt` files recursively and writes corresponding JSON files
under `data/tokenized/`. Run from the repository root:

```bash
python -m postag.tokenization
```

Existing tokenized outputs are protected; use `--overwrite` only when deliberately
regenerating them. This README is ignored by the `.txt` input filter.

Preserve original text and capitalization. Some source limitations, including
an unusually long lowercase sentence, are described in the
[tokenization guide](../../docs/TOKENIZATION.md).
