"""Small, hand-checkable pipeline tests; no network or corpus download needed."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from postag.corpus import read_document, sample_quotas, sample_rows, scan_corpus
from postag.evaluation import (coarse_pos, evaluate_clusters, file_sha256, load_dictionary,
                        lookup_key, partition_scores, reference_labels)
from postag import experiment
from postag import vectors
from postag.artifacts import load_transform
from scripts import fetch_dictionary
from postag.features import sent2features
from postag.vectors import FeatureEncoder, fit_reducer, select_features


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "tokens"
        self.source.mkdir()

    def write_document(self, name, sentences):
        path = self.source / name
        path.write_text(json.dumps([{"sentence": " ".join(tokens), "tokens": tokens}
                                    for tokens in sentences], ensure_ascii=False), encoding="utf-8")
        return path

    def test_sample_alignment_full_corpus_tf_and_context(self):
        self.write_document("a.json", [["Київ", ",", "гарний", "."], ["Він", "тут"]])
        self.write_document("b.json", [["Київ", "спить", "."]])
        frequencies, manifest = scan_corpus(self.source)
        self.assertEqual(frequencies["київ"], 2 / 9)
        self.assertEqual(frequencies["."], 2 / 9)
        rows, features = sample_rows(self.source, manifest, frequencies, limit=5, seed=11)
        self.assertEqual((rows, features), sample_rows(self.source, manifest, frequencies, 5, 11))
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({r["row_id"] for r in rows}), 5)
        self.assertEqual({r["document"] for r in rows}, {"a.json", "b.json"})
        for row, feature in zip(rows, features):
            sentence = read_document(self.source / row["document"])[0][row["sentence_index"]]["tokens"]
            self.assertEqual(feature, sent2features(sentence, frequencies)[row["token_index"]])
            self.assertEqual(row["token"], sentence[row["token_index"]])
        _, lexical = scan_corpus(self.source, "lexical")
        lexical_rows, lexical_features = sample_rows(self.source, lexical, frequencies, 0)
        index = next(i for i, r in enumerate(lexical_rows) if r["token"] == "гарний")
        self.assertEqual(lexical_features[index]["-1:word.lower()"], ",")
        self.assertEqual(lexical_features[index]["+1:word.lower()"], ".")

    def test_validation_and_changed_inputs(self):
        path = self.write_document("a.json", [["Я", "."]])
        frequencies, manifest = scan_corpus(self.source)
        self.write_document("a.json", [["Ти", "."]])
        with self.assertRaisesRegex(ValueError, "changed between passes"):
            sample_rows(self.source, manifest, frequencies)
        path.write_text('[{"sentence":"x","tokens":[null]}]')
        with self.assertRaisesRegex(ValueError, "invalid sentence"):
            scan_corpus(self.source)
        self.assertEqual(sample_quotas([0, 1, 20], 8, 42), [0, 1, 7])
        self.assertEqual(sample_quotas([0, 1, 20], 0, 42), [0, 1, 20])

    def test_persisted_transforms_and_small_input(self):
        features = sent2features(["Київ", "гарний", ".", "спить", "123"], {"київ": 0.1})
        self.assertNotIn("word.lower()", select_features(features[0], "context_tf"))
        self.assertNotIn("word.tf", select_features(features[0], "context"))
        self.assertFalse(any(k.startswith(("-1:", "+1:")) for k in select_features(features[1], "token")))
        for scaling in ("none", "std"):
            encoder = FeatureEncoder("context_tf", scaling)
            matrix = encoder.fit_transform(features)
            self.assertTrue(sparse.isspmatrix_csr(matrix))
            with threadpool_limits(limits=1):
                reducer, vectors, info = fit_reducer(matrix, 50, normalize=True)
            self.assertEqual(info["components"], 4)
            path = self.root / "transforms.joblib"
            joblib.dump((encoder, reducer), path)
            loaded_encoder, loaded_reducer = joblib.load(path)
            np.testing.assert_allclose(loaded_encoder.transform(features).toarray(), matrix.toarray())
            np.testing.assert_allclose(loaded_reducer.transform(matrix), vectors, atol=1e-12)
            new = sent2features(["Незнане", "слово"], {})
            self.assertEqual(loaded_reducer.transform(loaded_encoder.transform(new)).shape, (2, 4))
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit_reducer(sparse.csr_matrix([[1, 2]]), 30)

    def test_end_to_end_artifacts_and_output_protection(self):
        self.write_document("a.json", [["Київ", "гарний", "."], ["Кіт", "спить", "."]] * 8)
        dictionary = self.root / "dict.txt"
        dictionary.write_text("Київ noun:inanim:m:prop\nгарний adj:m\nкіт noun:anim:m\nспить verb:pres\n",
                              encoding="utf-8")
        output = self.root / "run"
        args = ["--input", str(self.source), "--output", str(output), "--sample-size", "30",
                "--variants", "token", "context_tf", "--components", "3", "--min-cluster-sizes", "3",
                "--min-samples", "2", "--dictionary", str(dictionary)]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiment.main(args), 0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
        results = json.loads((output / "results.json").read_text())
        self.assertEqual(len(results), 2)
        for result in results:
            assignments = np.load(output / result["id"] / "assignments.npz", allow_pickle=False)
            self.assertEqual(assignments["row_ids"].tolist(), [r["row_id"] for r in rows])
            self.assertEqual(len(assignments["labels"]), 30)
            self.assertIn("common_non_noise", result["evaluation"])
        self.assertTrue((output / "report.md").is_file())
        for directory in [output] + [p for p in output.rglob("*") if p.is_dir()]:
            self.assertTrue((directory / "README.md").is_file(), str(directory))
        self.assertIn("postag/features.py", manifest["code_sha256"])
        original = (output / "manifest.json").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiment.main(args), 1)
        self.assertEqual((output / "manifest.json").read_bytes(), original)

    def test_pre_package_encoder_loads_without_rewriting(self):
        features = sent2features(["Київ", "гарний", "."], {})
        encoder = FeatureEncoder("token")
        expected = encoder.fit_transform(features)
        path = self.root / "legacy.joblib"
        # Reproduce the actual module path stored by the original flat layout.
        with patch.dict(sys.modules, {"vectors": vectors}), patch.object(
                FeatureEncoder, "__module__", "vectors"):
            joblib.dump(encoder, path)
        original = path.read_bytes()
        loaded = load_transform(path)
        np.testing.assert_allclose(loaded.transform(features).toarray(), expected.toarray())
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(type(loaded).__module__, "postag.vectors")
        self.assertNotIn("vectors", sys.modules)


class EvaluationTests(unittest.TestCase):
    def test_dictionary_download_verifies_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            remote = directory / "remote.txt"
            remote.write_bytes(b"dictionary fixture")
            source = {"filename": "dictionary.txt", "url": remote.as_uri(),
                      "sha256": file_sha256(remote), "release": "test"}
            (directory / "source.json").write_text(json.dumps(source))
            with patch.object(fetch_dictionary, "DIRECTORY", directory), contextlib.redirect_stdout(io.StringIO()):
                fetch_dictionary.main()
                self.assertEqual((directory / "dictionary.txt").read_bytes(), remote.read_bytes())
                fetch_dictionary.main()  # Existing, verified file is reusable offline.
                (directory / "dictionary.txt").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "Existing dictionary checksum"):
                    fetch_dictionary.main()
                self.assertEqual((directory / "dictionary.txt").read_bytes(), b"corrupted")
                source["filename"] = "new.txt"
                source["sha256"] = "wrong"
                (directory / "source.json").write_text(json.dumps(source))
                with self.assertRaisesRegex(ValueError, "Downloaded dictionary checksum"):
                    fetch_dictionary.main()
                self.assertFalse((directory / "new.txt").exists())
                self.assertFalse(list(directory.glob(".download-*")))

    def test_dictionary_ambiguity_normalization_and_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dict.txt"
            path.write_text("мати noun:f:v_naz\n  матері noun:f:v_rod\nмати verb:inf\n"
                            "гарний adj:m:v_naz\n  гарний adj:m:v_zna\n"
                            "п'ять numr\nКиїв noun:m:prop\n", encoding="utf-8")
            lookup, provenance = load_dictionary(path, {"мати", "гарний", "п'ять", "київ"})
            self.assertEqual(lookup["мати"], {"NOUN", "VERB"})
            self.assertEqual(lookup["гарний"], {"ADJ"})
            self.assertEqual(lookup[lookup_key("ПʼЯТЬ")], {"NUM"})
            self.assertEqual(lookup["київ"], {"PROPN"})
            self.assertEqual(provenance["analysis_count"], 7)
            source = Path(folder) / "source.json"
            source.write_text(json.dumps({"sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dictionary(path, set(), source)
            source.write_text(json.dumps({"sha256": file_sha256(path)}))
            load_dictionary(path, set(), source)
            path.write_text("слово\tслово\tnoun:n\n", encoding="utf-8")
            lookup, _ = load_dictionary(path, {"слово"}, dictionary_format="lt")
            self.assertEqual(lookup["слово"], {"NOUN"})
            with self.assertRaisesRegex(ValueError, "columns"):
                load_dictionary(path, set())
        self.assertEqual(coarse_pos("noun:pron:pers"), "PRON")
        self.assertEqual(coarse_pos("adj:numr"), "NUM")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            coarse_pos("invented")

    def test_metrics_noise_and_coverage_by_hand(self):
        # Two mixed clusters: purity 3/4, not 1. A perfect partition is 1.
        truth = ["NOUN", "NOUN", "VERB", "VERB"]
        self.assertEqual(partition_scores(truth, [0, 0, 0, 1])["purity"], 0.75)
        perfect = partition_scores(truth, [5, 5, 2, 2])
        self.assertEqual((perfect["v_measure"], perfect["ari"], perfect["purity"]), (1, 1, 1))
        rows = [{"row_id": str(i), "token": token, "kind": kind} for i, (token, kind) in enumerate(
            [("кіт", "lexical"), ("кіт", "lexical"), ("мати", "lexical"),
             ("спить", "lexical"), ("???", "punctuation"), ("невідоме", "lexical")])]
        refs, coverage = reference_labels(rows, {"кіт": {"NOUN"}, "мати": {"NOUN", "VERB"},
                                                "спить": {"VERB"}})
        self.assertEqual(coverage["occurrences"], {"eligible": 3, "ambiguous": 1, "unmatched": 1, "nonlexical": 1})
        self.assertEqual(coverage["distinct_lookup_forms"]["eligible"], 2)
        score = evaluate_clusters(refs, [0, 0, 1, -1, 2, 3])
        self.assertEqual(score["eligible_count"], 3)
        self.assertEqual(score["assigned_count"], 2)
        self.assertAlmostEqual(score["eligible_noise_fraction"], 1 / 3)
        noise = evaluate_clusters(refs, [-1] * len(refs))
        self.assertIsNone(noise["non_noise"]["v_measure"])
        self.assertEqual(noise["non_noise"]["status"], "unavailable")
        self.assertEqual(noise["including_noise_as_one_label"]["count"], 3)
        self.assertIsNone(evaluate_clusters([], [])["eligible_noise_fraction"])
        with self.assertRaisesRegex(ValueError, "align"):
            evaluate_clusters(refs, [0])


if __name__ == "__main__":
    unittest.main()
