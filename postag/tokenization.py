"""Tokenize Ukrainian text, a file, or a directory without changing its spelling."""

import argparse
import codecs
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import tempfile


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "raw"
DEFAULT_OUTPUT = BASE_DIR / "data" / "tokenized"


@lru_cache(maxsize=1)
def get_pipeline():
    """Create the abbreviation-aware pipeline once, including for imported use."""
    try:
        from tokenize_uk.spacy import blank_pipeline
    except ImportError as error:
        raise RuntimeError(
            "Missing tokenizer dependency. Run: python -m pip install "
            "-r requirements/tokenization.txt"
        ) from error
    return blank_pipeline()


def tokenize_text(text: str) -> list[dict]:
    """Return original sentences and non-whitespace tokens; keep abbreviation dots."""
    if not text.strip():
        return []
    nlp = get_pipeline()
    # Whole books are processed without arbitrary cuts through sentences.
    nlp.max_length = max(nlp.max_length, len(text))
    # The upstream component assigns is_sent_start token by token. spaCy updates
    # sentence state on each assignment, which is costly for book-sized docs.
    # Apply the exact same library boundaries in one public from_array call.
    import numpy as np
    from spacy.attrs import SENT_START
    from tokenize_uk.pipeline import tokenize_sents_with_spans

    doc = nlp.make_doc(text)
    starts = {start for _, start, _ in tokenize_sents_with_spans(text)}
    flags = np.fromiter(
        (1 if i == 0 or token.idx in starts else -1 for i, token in enumerate(doc)),
        dtype=np.int64, count=len(doc),
    ).view(np.uint64)
    doc.from_array([SENT_START], flags.reshape((-1, 1)))
    return [
        {"sentence": sentence.text,
         "tokens": [token.text for token in sentence if not token.is_space]}
        for sentence in doc.sents
        if any(not token.is_space for token in sentence)
    ]


def split_sentences(text: str) -> list[str]:
    """Keep the original public helper available, using the library's boundaries."""
    return [item["sentence"] for item in tokenize_text(text)]


def read_text(path: Path, encoding: str) -> str:
    """Decode strictly and preserve original line endings."""
    with path.open(encoding=encoding, newline="") as stream:
        return stream.read()


def discover_files(source: Path, output: Path, recursive: bool) -> list[Path]:
    """Find text files deterministically, excluding symlinks and generated output."""
    source, output = source.resolve(), output.resolve()
    files = []
    for root, directories, names in os.walk(source, followlinks=False, onerror=raise_walk_error):
        parent = Path(root)
        directories[:] = sorted(
            name for name in directories
            if not (parent / name).is_symlink()
            and (parent / name).resolve() != output
        ) if recursive else []
        for name in sorted(names):
            path = parent / name
            if path.suffix.lower() == ".txt" and not path.is_symlink() and path.is_file():
                files.append(path)
    return sorted(files)


def raise_walk_error(error):
    raise error


def write_json(path: Path, result: list[dict], overwrite: bool) -> None:
    """Publish complete JSON atomically, with race-safe no-overwrite protection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".tokenization-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if overwrite:
            os.replace(temporary, path)
        else:
            # Unlike replace(), link() fails if the destination already exists.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_directory(source: Path, output: Path, encoding: str,
                      recursive: bool, overwrite: bool) -> int:
    source, output = source.resolve(), output.resolve()
    files = discover_files(source, output, recursive)
    if not files:
        print(f"No .txt files found in {source}", file=sys.stderr)
        return 1

    jobs = []
    destinations = {}
    for path in files:
        destination = output / path.relative_to(source).with_suffix(".json")
        # Case folding also catches collisions on case-insensitive filesystems.
        key = str(destination.resolve()).casefold()
        if key in destinations:
            raise ValueError(f"Output collision: {destinations[key]} and {path} -> {destination}")
        destinations[key] = path
        jobs.append((path, destination))

    successful = failed = conflicts = sentences = tokens = 0
    for path, destination in jobs:
        try:
            if not overwrite and os.path.lexists(destination):
                raise FileExistsError(f"Output exists: {destination}; use --overwrite to replace it")
            result = tokenize_text(read_text(path, encoding))
            write_json(destination, result, overwrite)
        except FileExistsError as error:
            conflicts += 1
            print(f"Conflict [{path}]: {error}", file=sys.stderr)
        except Exception as error:
            failed += 1
            print(f"Failed [{path}]: {error}", file=sys.stderr)
        else:
            successful += 1
            sentences += len(result)
            tokens += sum(len(item["tokens"]) for item in result)

    print(f"Successful: {successful}; failed: {failed}; conflicts: {conflicts}; "
          f"sentences: {sentences}; tokens: {tokens}", file=sys.stderr)
    return 1 if failed or conflicts else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text", help="Text to tokenize; write JSON to stdout")
    inputs.add_argument("--file", type=Path, help="Read one file; write JSON to stdout")
    inputs.add_argument("--directory", type=Path,
                        help=f"Input directory (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", type=Path,
                        help=f"Directory output (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--encoding", default="utf-8", help="Input encoding (default: utf-8)")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None,
                        help="Include subdirectories (default: enabled)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing JSON outputs")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    single = args.text is not None or args.file is not None
    if single and (args.output_dir is not None or args.recursive is not None or args.overwrite):
        parser.error("--output-dir, --recursive/--no-recursive and --overwrite require directory mode")
    try:
        codecs.lookup(args.encoding)
        if not single:
            source = (args.directory or DEFAULT_INPUT).resolve()
            output = (args.output_dir or DEFAULT_OUTPUT).resolve()
            if not source.is_dir():
                raise ValueError(f"Input directory does not exist or is not a directory: {source}")
            if source == output:
                raise ValueError("Input and output directories must differ")
            if output.exists() and not output.is_dir():
                raise ValueError(f"Output path is not a directory: {output}")
        get_pipeline()  # Fail once, before processing files, if dependencies are missing.
        if single:
            text = read_text(args.file, args.encoding) if args.file is not None else args.text
            print(json.dumps(tokenize_text(text), ensure_ascii=False, indent=2))
            return 0
        return process_directory(source, output, args.encoding,
                                 args.recursive is not False, args.overwrite)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
