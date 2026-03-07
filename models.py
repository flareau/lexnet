"""Domain models for lexical network data."""

from loader import DEFAULT_ENCODING, DEFAULT_SEPARATOR, load


class LexicalNetwork:
    """Main class for a lexical network."""

    def __init__(self, path, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
        self.path = path
        self.separator = separator
        self.encoding = encoding
        self.data = load(path=path, separator=separator, encoding=encoding)


class LexicalUnit:
    """Class for a lexical unit (node in the network)."""

    def __init__(self, node_id, ln):
        self.id = node_id
        self.ln = ln
        self.data = ln['nodes'].loc[node_id]
        self.features = ln['features'][ln['features'].node_id == node_id]

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

    def __init__(self, label_id, ln):
        self.id = label_id
        self.ln = ln
        self.data = ln['label_names'].loc[label_id]

class PropositionalForm:
    """Propositional forms for lexical units."""

    def __init__(self, form_id, ln):
        self.id = form_id
        self.ln = ln
        self.data = ln['propforms'].loc[form_id]

class Definition:
    """Definitions for lexical units."""

    def __init__(self, node_id, ln):
        self.id = node_id
        self.ln = ln
        self.data = ln['definitions'][ln['definitions'].node_id == node_id]

class LexicalFunction:
    """Lexical functions."""

    def __init__(self, lf_id, ln):
        self.id = lf_id
        self.ln = ln
        self.data = ln['lf_names'][ln['lf_names'].lf_id == lf_id]

class Example:
    """Examples for lexical units."""

    def __init__(self, node_id, ln):
        self.id = node_id
        self.ln = ln
        self.data = ln['examples'][ln['examples'].node_id == node_id]
