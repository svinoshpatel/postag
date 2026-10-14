import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from postag import tokenization as tokenizer


class TokenizationTests(unittest.TestCase):
    def test_bulk_boundaries_match_upstream(self):
        for text in ['Це проф. Петренко. Він у м. Київ.',
                     'Т. Г. Шевченко. Книги і т. ін. Далі.',
                     '\ufeff\t«…Звістка прийшла.\r\n\r\nЩе текст!',
                     '  Без крапки  ', 'Що?! Так… Авжеж.']:
            upstream = tokenizer.get_pipeline()(text)
            expected = [
                {'sentence': s.text, 'tokens': [t.text for t in s if not t.is_space]}
                for s in upstream.sents if any(not t.is_space for t in s)
            ]
            self.assertEqual(tokenizer.tokenize_text(text), expected)

    def test_abbreviations_and_initials(self):
        text = 'Це проф. Петренко. Він приїхав у м. Київ.'
        result = tokenizer.tokenize_text(text)
        self.assertEqual(len(result), 2)
        self.assertIn('проф.', result[0]['tokens'])
        self.assertIn('м.', result[1]['tokens'])
        self.assertEqual(len(tokenizer.tokenize_text('Т. Г. Шевченко написав вірш.')), 1)
        self.assertEqual(tokenizer.split_sentences('Книги і т. ін. Наступне речення.'),
                         ['Книги і т. ін.', 'Наступне речення.'])
        self.assertEqual(tokenizer.split_sentences('Він сказав: «Привіт!» Далі — 3.14.'),
                         ['Він сказав: «Привіт!»', 'Далі — 3.14.'])

    def test_preserves_characters(self):
        for text in [
            'Її сім’я, пʼять, п\'ять. «Київ» — Україна!',
            'Дата 27.09.2026, число 3.14. Що?! Так…',
            'Книги і т. ін. Наступне речення.',
            'Перший\r\nрядок без крапки', '', ' \n\t',
        ]:
            with self.subTest(text=text):
                result = tokenizer.tokenize_text(text)
                actual = ''.join(t for item in result for t in item['tokens'])
                self.assertEqual(''.join(actual.split()), ''.join(text.split()))
                for item in result:
                    self.assertIn(item['sentence'], text)
        self.assertEqual(tokenizer.tokenize_text('  \n'), [])

    def test_dates_and_final_fragment(self):
        result = tokenizer.tokenize_text('Дата 27.09.2026, число 3.14.')
        self.assertEqual(len(result), 1)
        self.assertIn('27.09.2026', result[0]['tokens'])
        self.assertIn('3.14', result[0]['tokens'])
        self.assertEqual(tokenizer.split_sentences('Без крапки'), ['Без крапки'])

    def test_length_limit(self):
        nlp = tokenizer.get_pipeline()
        original = nlp.max_length
        try:
            nlp.max_length = 5
            self.assertEqual(len(tokenizer.tokenize_text('Це довге речення.')), 1)
        finally:
            nlp.max_length = original


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'texts'
        self.source.mkdir()
        self.output = self.root / 'results'

    def write(self, name, text='Привіт, Київ!', encoding='utf-8'):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
        return path

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = tokenizer.main(list(args))
        return status, out.getvalue(), err.getvalue()

    def test_defaults_nested_outputs_and_overwrite(self):
        self.write('book.txt')
        self.write('nested/book.TXT')
        with patch.object(tokenizer, 'DEFAULT_INPUT', self.source), patch.object(
            tokenizer, 'DEFAULT_OUTPUT', self.output
        ):
            status, _, summary = self.run_cli()
            self.assertEqual(status, 0)
            self.assertIn('Successful: 2', summary)
            target = self.output / 'nested/book.json'
            original = target.read_bytes()
            self.assertTrue(self.output.joinpath('book.json').exists())
            self.assertEqual(json.loads(original)[0]['tokens'], ['Привіт', ',', 'Київ', '!'])
            self.write('nested/book.TXT', 'Інше речення.')
            self.assertEqual(self.run_cli()[0], 1)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(self.run_cli('--overwrite')[0], 0)
            self.assertNotEqual(target.read_bytes(), original)

    def test_discovery_exclusion_and_nonrecursive(self):
        self.write('a.txt')
        self.write('nested/b.txt')
        output = self.source / 'generated'
        output.mkdir()
        (output / 'ignored.txt').write_text('Не читати.')
        (self.source / 'linked.txt').symlink_to(self.source / 'a.txt')
        (self.source / 'linked-dir').symlink_to(self.source / 'nested', target_is_directory=True)
        self.assertEqual(len(tokenizer.discover_files(self.source, output, True)), 2)
        status, _, _ = self.run_cli('--directory', str(self.source), '--output-dir',
                                    str(output), '--no-recursive')
        self.assertEqual(status, 0)
        self.assertTrue((output / 'a.json').exists())
        self.assertFalse((output / 'nested').exists())

    def test_encoding_failure_continues(self):
        self.write('a.txt', 'Київ', encoding='cp1251')
        self.write('b.txt')
        args = ['--directory', str(self.source), '--output-dir', str(self.output)]
        status, _, summary = self.run_cli(*args)
        self.assertEqual(status, 1)
        self.assertIn('Successful: 1; failed: 1', summary)
        status, out, _ = self.run_cli('--file', str(self.source / 'a.txt'), '--encoding', 'cp1251')
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out)[0]['tokens'], ['Київ'])

    def test_empty_files_and_no_files(self):
        args = ['--directory', str(self.source), '--output-dir', str(self.output)]
        self.assertEqual(self.run_cli(*args)[0], 1)
        self.write('empty.txt', '')
        self.assertEqual(self.run_cli(*args)[0], 0)
        self.assertEqual(json.loads((self.output / 'empty.json').read_text()), [])

    def test_collisions_preflight(self):
        paths = [self.source / 'a.txt', self.source / 'a.TXT']
        with patch.object(tokenizer, 'discover_files', return_value=paths):
            status, _, message = self.run_cli('--directory', str(self.source),
                                             '--output-dir', str(self.output))
        self.assertEqual(status, 1)
        self.assertIn('Output collision', message)
        self.assertFalse(self.output.exists())

    def test_atomic_no_overwrite_and_cleanup(self):
        path = self.root / 'saved.json'
        path.write_text('original')
        with self.assertRaises(FileExistsError):
            tokenizer.write_json(path, [], False)
        self.assertEqual(path.read_text(), 'original')
        self.assertFalse(list(self.root.glob('.tokenization-*')))

    def test_invalid_options_and_configuration(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli('--text', 'Київ', '--no-recursive')
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.run_cli('--directory', str(self.source),
                                      '--output-dir', str(self.source))[0], 1)
        self.assertEqual(self.run_cli('--text', 'Київ', '--encoding', 'invalid-encoding')[0], 1)

    def test_original_line_endings(self):
        path = self.source / 'windows.txt'
        path.write_bytes('Київ\r\nУкраїна'.encode())
        self.assertEqual(tokenizer.read_text(path, 'utf-8'), 'Київ\r\nУкраїна')


if __name__ == '__main__':
    unittest.main()
