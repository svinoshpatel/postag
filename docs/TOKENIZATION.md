# Ukrainian text tokenization

All shell commands and Python examples below run from the repository root.

`postag/tokenization.py` divides Ukrainian text into sentences and then into tokens
(words, numbers, and punctuation). It uses `tokenize-uk` through its spaCy
integration, which provides Ukrainian abbreviation rules and preserves original
characters.

Capital letters stay capital letters. Punctuation is preserved. Abbreviation
periods remain attached: `проф.` is one token. No trained language model download
is needed. Tokenization is independent of the feature and clustering pipeline.

## Installation

Use the project virtual environment. From the project directory:

```bash
source .venv/bin/activate
python -m pip install -r requirements/tokenization.txt
```

If you do not have a virtual environment yet, create it first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/tokenization.txt
```

The code requires Python 3.9 or newer. Dependency versions are pinned for the
project's Python 3.9 environment; not every future Python release is necessarily
supported by those pins. `spacy` and its `thinc` dependency are pinned because
newer releases cannot install with this project's Python 3.9 setup. Installation
requires internet access; tokenization runs locally afterwards.

On platforms without a matching dependency wheel, installation may compile a
dependency such as Thinc and require platform C/C++ build tools. This project's
macOS Python 3.9 installation built Thinc successfully using its existing tools.

## Run with no arguments

```bash
python -m postag.tokenization
```

This reads all `.txt` files in `data/raw`, including subdirectories,
and writes JSON files to `data/tokenized`. Both default directories are located under the repository root, regardless of your current working directory.

Each input gets its own output:

```text
data/raw/book.txt         -> data/tokenized/book.json
data/raw/novels/book.txt  -> data/tokenized/novels/book.json
```

Existing outputs are protected. To intentionally regenerate them:

```bash
python -m postag.tokenization --overwrite
```

## Other ways to run

### Choose a directory

```bash
python -m postag.tokenization --directory texts --output-dir results
```

Explicit relative paths, such as `texts`, are relative to your current working
directory. Absolute paths also work. Quote paths containing spaces.

### Exclude subdirectories

```bash
python -m postag.tokenization --no-recursive
```

Recursion is enabled by default; `--recursive` makes that choice explicit.

### Tokenize text directly

```bash
python -m postag.tokenization --text "Це проф. Петренко. Він у Києві."
```

This prints JSON to the terminal instead of creating files.

### Tokenize one file

```bash
python -m postag.tokenization --file data/samples/Recording.txt
```

This also prints JSON. Use normal shell redirection if you want a file:

```bash
python -m postag.tokenization --file data/samples/Recording.txt > recording.json
```

Shell redirection follows your shell's overwrite behavior, not the directory
mode's `--overwrite` protection.

### Choose an encoding

```bash
python -m postag.tokenization --encoding cp1251
python -m postag.tokenization --file old-text.txt --encoding cp1251
```

UTF-8 is the default. Decoding is strict: invalid bytes produce an error rather
than disappearing or being replaced. Use the encoding you know the file uses.
A single directory run uses one encoding for all its input files.

Some corpus files start with a UTF-8 byte-order mark (BOM). With `utf-8`, that
character is preserved. To intentionally consume the initial BOM during decoding,
use `--encoding utf-8-sig`.

### Show all options

```bash
python -m postag.tokenization --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--directory PATH` | `data/raw` under the repository root | Process a directory. Also selected when no input option is given. |
| `--output-dir PATH` | `data/tokenized` under the repository root | Destination for directory results. |
| `--recursive` / `--no-recursive` | Enabled | Include or exclude subdirectories. |
| `--encoding NAME` | `utf-8` | Source-file encoding. |
| `--overwrite` | Disabled | Permit replacement of existing directory results. |
| `--text TEXT` | Unset | Process text and print JSON. |
| `--file PATH` | Unset | Process one file and print JSON. |

Choose at most one of `--text`, `--file`, and `--directory`. Directory-only options
(`--output-dir`, `--recursive`, `--no-recursive`, `--overwrite`) cannot be used
with `--text` or `--file`.

## Understanding the output

For `Олена живе в Києві.`:

```json
[
  {
    "sentence": "Олена живе в Києві.",
    "tokens": ["Олена", "живе", "в", "Києві", "."]
  }
]
```

The outer list contains one object per sentence. `sentence` is the original
text span selected by the library. `tokens` contains its non-whitespace tokens
in order. Empty or whitespace-only input produces `[]`.

Token boundaries follow the library's conventions. An abbreviation such as
`проф.` includes its period, while sentence-final punctuation is normally its
own token. Dates and decimal numbers may remain single tokens. Keeping
punctuation does not mean every punctuation character must be a separate token.

Original spelling, case, apostrophes, and quotes are preserved. Whitespace-only
tokens are omitted. The JSON is not a complete whitespace-preserving archive:
inter-sentence whitespace is not represented separately. Keep source files if
exact whole-document reconstruction is needed.

## Directory behavior and errors

- Files are processed in sorted order, one at a time. `.TXT` is accepted too.
- Symbolic-link files and subdirectories are skipped.
- The output directory is excluded from input discovery, even if placed inside
  the input directory. Input and output directories cannot be identical.
- Relative subdirectories are preserved, so separate folders can contain files
  with the same name. Colliding output paths are detected before processing;
  case-only differences are conservatively treated as collisions.
- Each JSON is written to a temporary file in its destination directory, then
  published only when complete. Without `--overwrite`, publication refuses an
  existing path even if it appeared during processing. This uses filesystem
  hard links; a filesystem that does not support them reports a write failure.
- A failure reading, decoding, tokenizing, or writing one file does not stop the
  remaining files. Existing-output conflicts also allow other files to proceed.
- Invalid global configuration, input traversal failures, or missing dependencies
  stop the run with an error.

A summary goes to stderr, for example:

```text
Successful: 2; failed: 0; conflicts: 0; sentences: 4; tokens: 18
```

Sentence and token totals include only successfully written results. Directory
mode does not print JSON to stdout. Single-input modes reserve stdout for JSON.

Exit codes:

- `0`: successful processing.
- `1`: processing/configuration failure, an existing-output conflict, or no
  matching files.
- `2`: invalid command-line arguments.

## How the code works

1. **`build_parser()`** defines command-line options. **`main()`** chooses the input
   mode, applies defaults, validates paths and encoding, and returns an exit code.
2. **`get_pipeline()`** creates the library pipeline. Its cache ensures that one
   process reuses a single pipeline across all files. Imports are delayed so
   `--help` works even before dependencies are installed.
3. **`read_text()`** decodes a file strictly and retains its original line endings.
4. **`tokenize_text()`** uses the pipeline's tokenizer and the library's
   `tokenize_sents_with_spans()` sentence rules. It applies sentence-start flags
   in one batch using spaCy's `Doc.from_array()` and NumPy, then iterates over
   `doc.sents` and collects original token text. This avoids the upstream
   component's slow token-by-token flag assignment on whole books. Tests compare
   the resulting boundaries with the original integration. It does not call
   `lower()` or normalize punctuation. **`split_sentences()`** provides the
   original public helper using those same sentence boundaries.
5. **`discover_files()`** finds eligible input files and excludes output and links.
6. **`process_directory()`** maps input paths to outputs, checks collisions, handles
   each file, and counts results.
7. **`write_json()`** saves readable UTF-8 JSON safely using a temporary file and
   atomic publication.

Use it from another Python script:

```python
from postag.tokenization import tokenize_text, split_sentences

result = tokenize_text("Це проф. Петренко. Він у Києві.")
for item in result:
    print(item["sentence"])
    print(item["tokens"])

sentences = split_sentences("Перше речення. Друге речення.")
```

## Limitations

Sentence detection uses linguistic rules, not perfect understanding. Abbreviations
at sentence endings, unusual initials, literary dialogue, and unconventional
punctuation can be ambiguous. Inspect a sample of your corpus before relying on
sentence counts as ground truth.

An observed corpus example beginning with `«…Звістка` produced a separate initial
sentence containing the opening quote and ellipsis (plus the leading BOM and
whitespace). All characters were retained, but this is not a desirable linguistic
boundary. No corpus-specific boundary overrides are applied.

The largest supplied corpus file contains 2,092,512 characters with no uppercase
letters or line breaks. The library treats it as one sentence. This script
preserves that source text and does not invent capitalization or alternative
sentence rules.

On the existing macOS Python 3.9 interpreter, a dependency emits an urllib3
`NotOpenSSLWarning` on import because Python uses LibreSSL. It goes to stderr;
the local tokenization checks still pass and JSON stdout is unaffected.

One complete file and its tokenized result are held in memory. The script raises
spaCy's length limit to accommodate the current document and does not cut books
at arbitrary character positions. Very large individual files can still require
substantial memory. Processing a directory does not load all files at once.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests exercise Ukrainian abbreviations, initials, character preservation,
dates, long-document limit handling, defaults, recursive discovery, output
protection, collisions, encodings, empty input, and error reporting. They create
outputs only in temporary directories.

Validation in the project's macOS Python 3.9 environment passed all 13 tests and
processed four real files: the smallest, median-by-size, and largest corpus files,
plus `data/samples/Recording.txt`. All non-whitespace characters were preserved in order.
The largest file took about 4.1 seconds including JSON writing; peak process
memory across the four-file run was about 249 MB. These are sample measurements,
not guarantees for other machines or texts. The full 200-file corpus was not run.

The approved design is recorded in [TOKENIZATIONPLAN.md](TOKENIZATIONPLAN.md).
Library documentation: [tokenize-uk](https://github.com/lang-uk/tokenize-uk).
