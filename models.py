"""Domain models for lexical network data."""

from loader import DEFAULT_ENCODING, DEFAULT_SEPARATOR, load


class LexicalNetwork:
    """Main class for a lexical network."""

    def __init__(self, path, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
        self.path = path
        self.separator = separator
        self.encoding = encoding
        self.data = load(path=path, separator=separator, encoding=encoding)

    def __getitem__(self, table):
        return self.data[table]


class LexicalUnit:
    """Class for a lexical unit (node in the network)."""

    def __init__(self, node_id, ln):
        self.id = node_id
        self.ln = ln
        self.data = ln['nodes'].loc[node_id]
        self.features = ln['features'].loc[node_id]
        self.wordforms = ln['forms'][ln['forms'].index == node_id]
        example_ids = ln['ex-rel'].loc[ln['ex-rel'].node_id == node_id, 'example_id']
        self.examples = ln['examples'].loc[example_ids.tolist()]


class LexicalEntry:
    """A lexical entry is a group of lexical units (copolysemes)."""

    def __init__(self, entry_id, ln):
        self.id = entry_id
        self.ln = ln
        self.data = ln['entries'].loc[entry_id]
        self.senses = ln['nodes'][ln['nodes'].entry_id == entry_id]


class GrammaticalFeature:
    """Grammatical features for lexical units."""

    def __init__(self, feature_id, ln):
        self.id = feature_id
        self.ln = ln
        self.data = ln['feature_names'].loc[feature_id]


class SemanticLabel:
    """Semantic labels for lexical units."""

    def __init__(self, semantic_label_id, ln):
        self.id = semantic_label_id
        self.ln = ln
        self.data = ln['label_names'].loc[semantic_label_id]


class PropositionalForm:
    """The propositional form for a lexical unit."""

    def __init__(self, node_id, ln):
        self.id = node_id
        self.ln = ln
        self.data = ln['propforms'].loc[node_id]


class Definition:
    """Definitions for lexical units."""

    def __init__(self, node_id, ln):
        self.id = node_id
        self.ln = ln
        self.data = ln['definitions'].loc[node_id]


class LexicalFunction:
    """Lexical functions."""

    def __init__(self, lexical_function_id, ln):
        self.id = lexical_function_id
        self.ln = ln
        self.data = ln['lf_names'].loc[lexical_function_id]


class Example:
    """An example from the lexical network."""

    def __init__(self, example_id, ln):
        self.id = example_id
        self.ln = ln
        self.data = ln['examples'].loc[example_id]
