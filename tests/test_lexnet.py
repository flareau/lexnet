import unittest

import lexnet


class TestPublicApi(unittest.TestCase):
    def test_domain_models_are_public(self):
        expected = {
            'Definition', 'Example', 'GrammaticalFeature', 'LexicalEntry',
            'LexicalFunction', 'LexicalNetwork', 'LexicalUnit',
            'PropositionalForm', 'SemanticLabel',
        }
        self.assertTrue(expected.issubset(set(lexnet.__all__)))

    def test_all_names_are_exported(self):
        for name in lexnet.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(lexnet, name))


if __name__ == '__main__':
    unittest.main()
