# Tokenization refactoring plan

All shell commands and Python examples below run from the repository root.

This document describes the proposed implementation. Creating this plan does not
change `postag/tokenization.py` or install dependencies.

## Requirements and agreed decisions

- Split Ukrainian text into sentences, then words and punctuation tokens.
- Preserve original capitalization and all punctuation characters.
- Preserve original apostrophe and quotation characters.
- Keep abbreviation dots attached: `проф.` is one token, not `проф` and `.`.
- Process an entire directory in one run, including nested directories by default.
- Allow running with no arguments by providing sensible defaults.
- Keep the existing CRF training code outside the scope of this refactor.

## 1. Verify and declare dependencies

Verify the released version of `tokenize-uk` that exposes
`tokenize_uk.spacy.blank_pipeline()` and its compatibility with the project's
Python version. Declare the tested dependency and its spaCy extra in a dedicated
`requirements/tokenization.txt` file.

Use the spaCy integration because its documented behavior preserves exact token
text and character offsets. The plain `tokenize_words()` interface normalizes
some apostrophe and quotation characters.

Reference: https://github.com/lang-uk/tokenize-uk/blob/master/README.md

## 2. Replace custom tokenization rules

Initialize `blank_pipeline()` once per invocation and reuse it for every file.
Replace the current sentence-boundary and token regular expressions with:

- `doc.sents` for sentence boundaries;
- `sentence.text` for original sentence text;
- `token.text` for tokens, excluding only whitespace tokens.

Retain the existing result format:

```json
[
  {
    "sentence": "Олена живе в Києві.",
    "tokens": ["Олена", "живе", "в", "Києві", "."]
  }
]
```

Write JSON in UTF-8 with `ensure_ascii=False` and an indentation of two spaces.
Keep `tokenize_text()` usable from Python code, independently of the CLI.

## 3. Handle abbreviations through the library

Use the library's Ukrainian abbreviation and sentence-boundary rules. Keep
abbreviations such as `проф.` and `м.` intact, including their dots. Do not add
a postprocessing step that splits abbreviation dots.

Verify examples including:

- `Це проф. Петренко. Він приїхав у м. Київ.`
- `Т. Г. Шевченко написав вірш.`
- Multi-part abbreviations such as `т. д.` and `т. ін.`.
- Abbreviations occurring at the end of a sentence.
- Dates, decimals, quotes, and sentences without final punctuation.

Do not assume every ambiguous sentence boundary is solved by the library.
Document observed limitations; add narrowly scoped exceptions only if corpus
examples demonstrate that they are necessary.

## 4. CLI options and defaults

Resolve default directories relative to the repository root, so running the script
from another working directory still selects the same input corpus and output.
Resolve explicitly supplied relative paths against the current working directory.

| Option | Default | Behavior |
| --- | --- | --- |
| Input mode | Directory mode | Used when none of `--text`, `--file`, or `--directory` is supplied. |
| `--directory PATH` | `<repository root>/data/raw` | Directory containing source texts. |
| `--output-dir PATH` | `<repository root>/data/tokenized` | Destination for directory-mode JSON results. |
| `--encoding NAME` | `utf-8` | Source-file encoding; decoding is strict. |
| `--recursive` / `--no-recursive` | Recursive | Include or exclude nested input directories. |
| `--overwrite` | Disabled | Explicitly permit replacement of existing outputs. |
| `--text TEXT` | Unset | Tokenize supplied text and print JSON to stdout. |
| `--file PATH` | Unset | Tokenize one file and print JSON to stdout. |

The three explicit input options are mutually exclusive. Apply the default
directory only after argument parsing when no explicit input mode was selected.
Directory-specific options should produce a clear argument error when explicitly
used with `--text` or `--file` instead of being silently ignored.

Process files with a `.txt` suffix, case-insensitively, in deterministic sorted
order. Keep this selection rule fixed initially rather than adding an unnecessary
configuration option.

Examples:

```bash
# Process the default corpus recursively with all default settings.
python -m postag.tokenization

# Use custom directories; other options keep their defaults.
python -m postag.tokenization --directory texts --output-dir results

# Process only files directly inside the default corpus directory.
python -m postag.tokenization --no-recursive

# Explicitly regenerate existing results.
python -m postag.tokenization --overwrite

# Read files in a different known encoding.
python -m postag.tokenization --encoding cp1251

# Preserve existing single-input workflows.
python -m postag.tokenization --text "Це проф. Петренко. Він у Києві."
python -m postag.tokenization --file data/samples/Recording.txt
```

## 5. Directory processing and output safety

- Validate that the input directory exists and is a directory.
- Reject identical input and output directories.
- Exclude the output directory and its descendants from input discovery.
- Do not follow symbolic links during directory traversal.
- Process one source file at a time instead of loading the whole corpus.
- Preserve relative subdirectories and produce one JSON file per source:
  `data/raw/novels/book.txt` becomes `data/tokenized/novels/book.json`.
- Detect output-path collisions before processing rather than silently replacing
  one source's result with another's.
- Without `--overwrite`, report existing output files as conflicts, leave them
  intact, and continue processing other files.
- Create destination parent directories as needed.
- Write to a temporary file in the destination directory, then publish the
  completed file. Ensure publication respects overwrite protection and cleans up
  temporary files on failure.

## 6. Errors, encoding, and large texts

Report source paths and actionable error messages for decoding, reading,
tokenization, and writing failures. Continue with other files when a failure is
isolated to one input. Fail immediately for invalid global configuration or an
unavailable dependency.

Do not silently replace undecodable characters or guess encodings. Use
`--encoding` when the corpus has a known encoding other than UTF-8.

Account for spaCy's document-length limit before processing long books. For this
tokenization-only pipeline, explicitly adjust the limit to accommodate the
current document, then validate memory use on a representative large file.
Do not split at arbitrary character positions or individual newlines, since
these may cut sentences in half. If whole-document processing proves impractical,
design and verify a boundary-aware strategy before adding chunking.

Print a final directory-run summary to stderr with successful, failed, and
conflicting file counts, plus sentence and token totals for successful files.
Return a nonzero exit status for failures, conflicts, or no matching input files.
Single-input modes should keep stdout reserved for JSON.

## 7. Verification

Add focused automated tests for:

- Abbreviations inside sentences and at sentence endings, plus initials.
- Attached abbreviation dots, exact capitalization, apostrophes, and quotes.
- Decimal numbers, dates, repeated punctuation, and final fragments.
- Empty and whitespace-only inputs.
- Preservation of all non-whitespace characters in order across the output tokens.
- No-argument defaults and explicit CLI overrides.
- Nested directories and duplicate basenames in different subdirectories.
- Output exclusion, existing-output protection, and explicit overwrite behavior.
- Encoding failures, continued processing, summary counts, and exit statuses.

Run a representative sample of the actual corpus, including a large text, and
inspect sentence boundaries before treating the refactor as complete. Keep
generated validation outputs in a temporary directory.

## 8. Documentation and delivery

Document dependency installation, default paths, CLI examples, output format,
abbreviation token conventions, encoding selection, and observed limitations.
Explain the final code and report the checks performed after implementation.

Implementation begins only after the user approves this plan.
