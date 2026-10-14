"""Feature behavior checks; no CRF or tokenizer dependencies are required."""
import unittest
from unittest.mock import patch

from postag import features as main


class FeatureTests(unittest.TestCase):
    def test_ukrainian_features_and_context(self):
        sentence = ['Київ', 'СЛОВО', '123']
        frequencies = main.build_word_frequencies([sentence])
        features = main.word2features(sentence, 1, frequencies)
        self.assertEqual(features['word.prefix3'], 'сло')
        self.assertEqual(features['word.suffix4'], 'лово')
        self.assertEqual(features['word.length'], 5)
        self.assertTrue(features['word.isupper()'])
        self.assertTrue(features['-1:word.istitle()'])
        self.assertTrue(features['+1:word.isdigit()'])
        for prefix, word in [('-1:', 'Київ'), ('', 'СЛОВО'), ('+1:', '123')]:
            for name, value in main.token_features(word, frequencies).items():
                self.assertEqual(features[prefix + name], value)
        self.assertNotIn('BOS', features)
        self.assertNotIn('EOS', features)

    def test_tf_normalization_and_unseen_tokens(self):
        frequencies = main.build_word_frequencies([['Київ', 'київ'], ['.']])
        self.assertEqual(frequencies, {'київ': 2 / 3, '.': 1 / 3})
        self.assertEqual(main.word2features(['КИЇВ'], 0, frequencies)['word.tf'], 2 / 3)
        self.assertEqual(main.word2features(['Львів'], 0, frequencies)['word.tf'], 0.0)
        self.assertEqual(main.build_word_frequencies([[]]), {})

    def test_boundaries_short_tokens_and_empty_sentences(self):
        features = main.word2features(['Я'], 0, {})
        self.assertTrue(features['BOS'])
        self.assertTrue(features['EOS'])
        self.assertEqual(features['word.suffix4'], 'я')
        self.assertFalse(any(key.startswith(('-1:', '+1:')) for key in features))
        self.assertEqual(main.sent2features([], {}), [])
        for index in (-1, 1):
            with self.assertRaises(IndexError):
                main.word2features(['Я'], index, {})

    def test_annotations_never_affect_features_or_frequencies(self):
        plain = ['Київ', 'гарний']
        annotated = [('Київ', 'PROPN', 'B-LOC'), ('гарний', 'ADJ', 'O')]
        changed = [('Київ', 'OTHER', 'O'), ('гарний', 'OTHER', 'B-LOC')]
        frequencies = main.build_word_frequencies([plain])
        for sentence in (annotated, changed):
            self.assertEqual(main.build_word_frequencies([sentence]), frequencies)
            self.assertEqual(main.sent2features(sentence, frequencies),
                             main.sent2features(plain, frequencies))

    def test_custom_feature_applies_to_every_position(self):
        with patch.dict(main.WORD_FEATURES, {'word.has_hyphen': lambda word: '-' in word}):
            features = main.word2features(['будь-який', 'текст', 'по-нашому'], 1, {})
        self.assertTrue(features['-1:word.has_hyphen'])
        self.assertFalse(features['word.has_hyphen'])
        self.assertTrue(features['+1:word.has_hyphen'])


if __name__ == '__main__':
    unittest.main()
