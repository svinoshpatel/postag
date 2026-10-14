# Data

| Directory | Role |
| --- | --- |
| [raw/](raw/README.md) | Original 200-document Ukrainian text corpus. |
| [tokenized/](tokenized/README.md) | Sentence/token JSON consumed by experiments. |
| [samples/](samples/README.md) | Standalone sample text for tokenizer checks. |
| [dict_uk/](dict_uk/README.md) | Pinned dictionary provenance and downloaded POS reference. |

The processing flow is `raw/` → tokenizer → `tokenized/` → experiment → `runs/`.
The dictionary is used only for evaluation, never as input features. Commands are
run from the repository root. Generated vectors and scores belong in `runs/`,
not alongside corpus inputs.

Reorganization preserved corpus and dictionary contents. Original raw filenames
are retained, including their existing unusual encoding appearance.
