# Sample text

[Recording.txt](Recording.txt) is the original standalone text sample, moved
from the repository root without changing its contents. It is separate from the
default raw corpus.

Tokenize it from the repository root:

```bash
python -m postag.tokenization --file data/samples/Recording.txt
```

Single-file mode prints JSON to stdout. For directory behavior and encoding
options, see the [tokenization guide](../../docs/TOKENIZATION.md).
