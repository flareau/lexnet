import os
import unittest

from loader import load


class TestRealDataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.environ.get('LEXNET_DATA_PATH')
        if not cls.data_path:
            raise unittest.SkipTest('Set LEXNET_DATA_PATH to run real-data integration tests')
        cls.ln = load(cls.data_path)

    def test_core_tables_exist(self):
        required = {
            'nodes', 'entries', 'copolysemy', 'features', 'forms',
            'labels', 'propforms', 'lfs', 'definitions',
            'feature_names', 'form_names', 'label_names', 'lf_names',
            'examples', 'ex-rel'
        }
        self.assertTrue(required.issubset(set(self.ln.keys())))

    def test_expected_indexes(self):
        self.assertEqual(self.ln['nodes'].index.name, 'node_id')
        self.assertEqual(self.ln['entries'].index.name, 'entry_id')
        self.assertEqual(self.ln['forms'].index.name, 'node_id')
        self.assertEqual(self.ln['labels'].index.name, 'node_id')
        self.assertEqual(self.ln['definitions'].index.name, 'node_id')
        self.assertEqual(self.ln['examples'].index.name, 'example_id')
        self.assertEqual(self.ln['feature_names'].index.name, 'feature_id')
        self.assertEqual(self.ln['form_names'].index.name, 'wordform_feature_id')
        self.assertEqual(self.ln['label_names'].index.name, 'semantic_label_id')
        self.assertEqual(self.ln['lf_names'].index.name, 'lexical_function_id')

    def test_key_columns_present(self):
        self.assertIn('entry_id', self.ln['nodes'].columns)
        self.assertIn('lexname', self.ln['nodes'].columns)
        self.assertIn('std_name', self.ln['nodes'].columns)
        self.assertIn('features', self.ln['features'].columns)
        self.assertIn('propform', self.ln['propforms'].columns)

    def test_entry_superscripts_are_nullable_integers(self):
        superscripts = self.ln['entries']['superscript']
        self.assertEqual(str(superscripts.dtype), 'Int64')
        self.assertTrue(superscripts.dropna().map(lambda value: isinstance(value, int)).all())

    def test_single_record_tables_have_unique_indexes(self):
        tables = {
            'nodes', 'entries', 'features', 'labels', 'propforms',
            'definitions', 'examples', 'feature_names', 'form_names',
            'label_names', 'lf_names',
        }
        for table in tables:
            with self.subTest(table=table):
                self.assertTrue(self.ln[table].index.is_unique)

    def test_foreign_keys_resolve(self):
        relations = [
            ('nodes.entry_id', self.ln['nodes']['entry_id'], self.ln['entries'].index),
            ('features.node_id', self.ln['features'].index, self.ln['nodes'].index),
            ('forms.node_id', self.ln['forms'].index, self.ln['nodes'].index),
            ('labels.node_id', self.ln['labels'].index, self.ln['nodes'].index),
            ('labels.semantic_label_id', self.ln['labels']['semantic_label_id'], self.ln['label_names'].index),
            ('propforms.node_id', self.ln['propforms'].index, self.ln['nodes'].index),
            ('definitions.node_id', self.ln['definitions'].index, self.ln['nodes'].index),
            ('copolysemy.source_node_id', self.ln['copolysemy']['source_node_id'], self.ln['nodes'].index),
            ('copolysemy.target_node_id', self.ln['copolysemy']['target_node_id'], self.ln['nodes'].index),
            ('lfs.source_node_id', self.ln['lfs']['source_node_id'], self.ln['nodes'].index),
            ('lfs.target_node_id', self.ln['lfs']['target_node_id'], self.ln['nodes'].index),
            ('lfs.lexical_function_id', self.ln['lfs']['lexical_function_id'], self.ln['lf_names'].index),
            ('ex-rel.node_id', self.ln['ex-rel']['node_id'], self.ln['nodes'].index),
            ('ex-rel.example_id', self.ln['ex-rel']['example_id'], self.ln['examples'].index),
        ]
        for relation, child_keys, parent_keys in relations:
            with self.subTest(relation=relation):
                self.assertTrue(set(child_keys).issubset(set(parent_keys)))


if __name__ == '__main__':
    unittest.main()
