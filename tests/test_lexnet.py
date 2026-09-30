import unittest

import lexnet


class TestPublicApi(unittest.TestCase):
    def test_all_names_are_exported(self):
        for name in lexnet.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(lexnet, name))


if __name__ == '__main__':
    unittest.main()
