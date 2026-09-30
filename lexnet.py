"""Compatibility module exposing loader functions, models, and a test-mode CLI."""

import argparse

from loader import (
    DEFAULT_COLUMNS,
    DEFAULT_DATA_SOURCES,
    DEFAULT_ENCODING,
    DEFAULT_SEPARATOR,
    load,
    load_csv,
    load_feature_names,
    load_form_names,
    load_label_names,
    load_lf_names,
    load_xml,
    normalize_propforms,
    print_imported,
    std_name,
    to_list,
)
from models import (
    Definition,
    Example,
    GrammaticalFeature,
    LexicalEntry,
    LexicalFunction,
    LexicalNetwork,
    LexicalUnit,
    PropositionalForm,
    SemanticLabel,
)


__all__ = [
    'DEFAULT_COLUMNS',
    'DEFAULT_DATA_SOURCES',
    'DEFAULT_ENCODING',
    'DEFAULT_SEPARATOR',
    'Definition',
    'Example',
    'GrammaticalFeature',
    'LexicalEntry',
    'LexicalFunction',
    'LexicalNetwork',
    'LexicalUnit',
    'PropositionalForm',
    'SemanticLabel',
    'load',
    'load_csv',
    'load_feature_names',
    'load_form_names',
    'load_label_names',
    'load_lf_names',
    'load_xml',
    'normalize_propforms',
    'print_imported',
    'std_name',
    'to_list',
    'smoke_test',
]


def smoke_test(path, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
    """Load a lexical network and print all tables."""
    ln = load(path=path, separator=separator, encoding=encoding)

    print(f'\nSample from {len(ln)} tables:\n')
    for table, data in ln.items():
        print(f'\x1b[0;34m{table}\x1b[0m')
        print(data)
        print()

    return ln

if __name__ == '__main__':
    print('\x1b[0;31mRunning in test mode\x1b[0m')

    parser = argparse.ArgumentParser(description='Run a smoke test for lexical network loading.')
    parser.add_argument('-d', '--data', required=True, help='Path to the data folder')
    parser.add_argument('-s', '--separator', default=DEFAULT_SEPARATOR, help='CSV separator')
    parser.add_argument('-e', '--encoding', default=DEFAULT_ENCODING, help='character encoding')
    args = parser.parse_args()

    smoke_test(path=args.data, separator=args.separator, encoding=args.encoding)
