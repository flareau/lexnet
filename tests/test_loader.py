import tempfile
import unittest
from pathlib import Path

from loader import load_label_model, load_label_names, to_list


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

    def test_semantic_label_model_preserves_classes_edges_and_memberships(self):
        with tempfile.TemporaryDirectory() as path:
            file = Path(path) / 'labels.xml'
            file.write_text('''
                <model>
                  <class id="c0" name="ROOT" status="1" semfield="0" inheritancetype="0" comment="">
                    <class id="c1" name="CHILD" status="1" semfield="1" inheritancetype="1" comment="Note">
                      <instance id="sl1" name="Label One" status="1" derivation="A0" acttype="0" comment="" />
                    </class>
                  </class>
                </model>
            ''', encoding='utf8')
            labels, classes, edges, memberships = load_label_model(file.name, path)

        self.assertEqual(labels.iloc[0]['derivation'], 'A0')
        self.assertEqual(classes['semantic_class_id'].tolist(), ['c0', 'c1'])
        self.assertEqual(edges.iloc[0].to_dict(), {
            'parent_class_id': 'c0', 'child_class_id': 'c1',
        })
        self.assertEqual(memberships.iloc[0].to_dict(), {
            'semantic_class_id': 'c1', 'semantic_label_id': 'sl1',
        })


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
