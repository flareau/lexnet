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
        self.assertEqual(self.ln['examples'].index.name, 'ex_id')
        self.assertEqual(self.ln['feature_names'].index.name, 'feature_id')
        self.assertEqual(self.ln['form_names'].index.name, 'form_id')
        self.assertEqual(self.ln['label_names'].index.name, 'label_id')
        self.assertEqual(self.ln['lf_names'].index.name, 'lf_id')

    def test_key_columns_present(self):
        self.assertIn('entry_id', self.ln['nodes'].columns)
        self.assertIn('lexname', self.ln['nodes'].columns)
        self.assertIn('std_name', self.ln['nodes'].columns)
        self.assertIn('features', self.ln['features'].columns)
        self.assertIn('propform', self.ln['propforms'].columns)


if __name__ == '__main__':
    unittest.main()
