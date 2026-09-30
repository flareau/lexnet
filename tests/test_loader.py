import tempfile
import unittest
from pathlib import Path

from loader import load_label_names, to_list


class TestLabelNames(unittest.TestCase):
    def load_xml(self, content):
        with tempfile.TemporaryDirectory() as path:
            file = Path(path) / 'labels.xml'
            file.write_text(content, encoding='utf8')
            return load_label_names(file.name, path)

    def test_identical_duplicate_ids_are_collapsed(self):
        labels = self.load_xml('''
            <labels>
              <instance id="sl1" name="Label One" />
              <instance id="sl1" name="Label One" />
              <instance id="sl2" name="Label Two" />
            </labels>
        ''')

        self.assertEqual(labels['semantic_label_id'].tolist(), ['sl1', 'sl2'])

    def test_conflicting_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'sl1'):
            self.load_xml('''
                <labels>
                  <instance id="sl1" name="First name" />
                  <instance id="sl1" name="Second name" />
                </labels>
            ''')


class TestListParsing(unittest.TestCase):
    def test_parses_list(self):
        self.assertEqual(to_list('(one,two)'), ['one', 'two'])

    def test_parses_empty_and_missing_values(self):
        self.assertEqual(to_list('()'), [])
        self.assertEqual(to_list(None), [])

    def test_rejects_malformed_list(self):
        with self.assertRaisesRegex(ValueError, 'Invalid LexNet list'):
            to_list('one,two')


if __name__ == '__main__':
    unittest.main()
