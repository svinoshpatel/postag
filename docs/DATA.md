# Gold corpus: provenance and split statistics

## Dataset and revision

The project uses [UD Ukrainian-IU](https://github.com/UniversalDependencies/UD_Ukrainian-IU), developed by the Institute for Ukrainian. The files were imported on **2026-11-07** from the existing local checkout `UD_Ukrainian-IU`.

- Exact commit: [`bcb3e6bd53a68cf1c680528ee86f0f6102d5085e`](https://github.com/UniversalDependencies/UD_Ukrainian-IU/commit/bcb3e6bd53a68cf1c680528ee86f0f6102d5085e).
- Git description: `r2.17-1-gbcb3e6b` (one commit after tag `r2.17`, not the exact tagged release).
- Commit subject: `Fixed obl-should-be-nmod.`
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Retain attribution to the corpus creators and observe its non-commercial/share-alike terms when distributing data or derivatives. See upstream README and license for details.

All three source files matched their committed Git blobs before import. At the user's explicit request, the files were moved out of that checkout and renamed in this project. Only paths changed: bytes, annotations, sentence order and official split membership were preserved. The sibling checkout consequently shows the source files as deleted; its Git history remains available.

## Split and overall statistics

The upstream README describes a manual, approximately **75% training / 10% development / 15% test** split, balancing genre and complexity. Unequal sizes are intentional. Its summary table is older than this snapshot; the counts below were measured from the imported bytes.

| Project file | Upstream file | Sentences | Integer-word tokens | Token share | Bytes |
|---|---|---:|---:|---:|---:|
| `data/raw/gold/train_gold.conllu` | `uk_iu-ud-train.conllu` | 5,521 | 92,927 | 75.70% | 13,165,577 |
| `data/raw/gold/val_gold.conllu` | `uk_iu-ud-dev.conllu` | 673 | 12,606 | 10.27% | 1,815,776 |
| `data/raw/gold/test_gold.conllu` | `uk_iu-ud-test.conllu` | 898 | 17,217 | 14.03% | 2,506,721 |
| **Total** | | **7,092** | **122,750** | **100%** | **17,488,074** |

Sentences were counted from `# sent_id =` comments. Tokens count positive integer-ID syntactic-word rows, including punctuation; multiword-token range rows and decimal empty nodes are excluded. These are structural counts, not label-distribution analysis or model evaluation.

Use training to teach the model, validation to select settings, and test only for the frozen final comparison. **Do not merge and randomly repartition these files.** The upstream README notes that some large documents span splits: this benchmark is not a guarantee of entirely unseen documents/authors. Literature source separation and tokenizer compatibility remain separate checks; import alone does not complete corpus readiness. See [PLAN.md](../PLAN.md#corpus-readiness-and-experiment-gate) and the authoritative [gold contract](ARCH.md#gold-conll-u-input-contract).

## Content checksums

SHA-256 was checked before and after relocation. Preserve these original files unchanged.

| Project filename | SHA-256 |
|---|---|
| `train_gold.conllu` | `ded6560c066765e3d8b7aad5f3e5bfd63494b6fb26f521c0019cae56bbecdab4` |
| `val_gold.conllu` | `fc68644aeb86766935f1c3aa0e8baf42cc6325690803631e81e17ed8db79f436` |
| `test_gold.conllu` | `5538dec31c89b0cd0e5141d8a9c6bdc0099103eab3c5c6b9078ab95e67be1286` |

## Configuration

Use these paths in the `data` section of your experiment YAML (see [README](../README.md#create-experimentyml) for the other settings):

```yaml
data:
  train_gold: data/raw/gold/train_gold.conllu
  val_gold: data/raw/gold/val_gold.conllu
  test_gold: data/raw/gold/test_gold.conllu
  ud_release: "r2.17-1-gbcb3e6b"
  literature_dir: data/raw/literature
  source_encoding: utf-8
```

The release string identifies this snapshot; exact commit and file checksums above disambiguate it. The pipeline also records input checksums automatically for each run. Gold is UTF-8; `source_encoding` controls literature decoding and must match the prepared books.
