"""Import data from a Lexical Network into pandas dataframes."""

import pandas as pd
import xml.etree.ElementTree as ET
import re
from pathlib import Path

DEFAULT_SEPARATOR = '\t'
DEFAULT_ENCODING = 'utf8'

DEFAULT_DATA_SOURCES = {
    'nodes': '01-lsnodes.csv',
    'entries': '02-lsentries.csv',
    'copolysemy_names': '03-lscopolysemy-model.xml',
    'copolysemy': '04-lscopolysemy-rel.csv',
    'feature_names': '05-lsgramcharac-model.xml',
    'features': '06-lsgramcharac-rel.csv',
    'form_names': '07-lswordform-model.xml',
    'forms': '08-lswordforms.csv',
    'label_names': '09-lssemlabel-model.xml',
    'labels': '10-lssemlabel-rel.csv',
    'propforms': '11-lspropform-rel.csv',
    # DTD ignored: '12-definition.dtd'
    'definitions': '13-lsdef.csv',
    'lf_names': '14-lslf-model.xml',
    'lfs': '15-lslf-rel.csv',
    'examples': '17-lsex.csv',
    'ex-rel': '18-lsex-rel.csv',
}

DEFAULT_COLUMNS = {
    'nodes': ['node_id', 'entry_id', 'lexnum', 'node_status', 'node_%', 'node_update_date', 'node_update_time', 'lexname'],
    'entries': ['entry_id', 'addtoname', 'entry_name', 'subscript', 'superscript', 'entry_status', 'entry_%'],
    'copolysemy': ['cp_source', 'cp_target', 'cp_type', 'cp_subtype'],
    'features': ['node_id', 'usage', 'usagevars', 'POS', 'ph_str', 'embeddedlex', 'features', 'featuresvars'],
    'forms': ['node_id', 'features', 'signifier'],
    'labels': ['node_id', 'label', 'label_%'],
    'propforms': ['node_id', 'propform', 'tildevalue', 'propform_confid', 'actants'],
    'lfs': ['source_id', 'lf_id', 'target_id', 'form', 'separator', 'merged', 'frame', 'constraint', 'position'],
    'definitions': ['node_id', 'def_XML', 'def_HTML'],
    'examples': ['ex_id', 'source', 'status', 'content', 'title', 'authors', 'location', 'date'],
    'ex-rel': ['node_id', 'ex_id', 'occurrence', 'position', '%'],
}


def print_imported(data, file, items=None):
    if not items:
        items = 'items'
    print(f"Imported {len(data)} {items} from {file}")


def load_csv(file, path, columns=None, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
    data = pd.read_csv(Path(path) / file, sep=separator, encoding=encoding, header=0, names=columns)
    print_imported(data, file)
    return data


def load_xml(file, path, encoding=DEFAULT_ENCODING):
    # ET.parse() doesn't support encoding specification
    with open(Path(path) / file, 'r', encoding=encoding) as f:
        xml = ET.fromstring(f.read())
    return xml


def load_feature_names(file, path, encoding=DEFAULT_ENCODING):
    xml = load_xml(file=file, path=path, encoding=encoding)
    features = xml.iter('characteristic')
    feature_names = pd.DataFrame([{'feature_id': f.get('id'), 'name': f.get('name')} for f in features])
    print_imported(feature_names, file, items='grammatical features')
    return feature_names


def load_lf_names(file, path, encoding=DEFAULT_ENCODING):
    xml = load_xml(file=file, path=path, encoding=encoding)
    tags = xml.findall('group/family/lexicalfunction')
    lf_names = pd.DataFrame([
        {'lf_id': tag.get('id'), 'lf_name': tag.get('name'), 'type': tag.get('linktype')}
        for tag in tags
    ])
    print_imported(lf_names, file)
    return lf_names


def std_name(lexname):
    """Return the standard name for a lexical unit."""
    lexname = re.sub(r"<span class='namingform'>([^<]+)</span>", r'\1', lexname)
    lexname = re.sub(r"<span class='(sup|sub)'>([^<]+)</span>", r'_\2', lexname)
    lexname = re.sub(r"<span class='num'>([^<]+)</span>", r'#\1', lexname)
    return lexname


def normalize_propforms(propform):
    """Fix common bugs in propositional forms."""
    propform = re.sub(r"'", r'’', propform)
    propform = re.sub(r' +,', r',', propform)
    propform = re.sub(r'(\w)~', r'\1 ~', propform)
    propform = re.sub(r'~(\w)', r'~ \1', propform)
    propform = re.sub(r'\s\s+', r' ', propform)
    return propform.strip()


def to_list(text):
    """Convert a LN list such as '(ls:fr:gc:26,ls:fr:gc:73)' to a Python list."""
    if isinstance(text, str):
        assert text[0] == '('
        assert text[-1] == ')'
        return text[1:-1].split(',')
    return []


def load(path, sources=None, columns=None, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
    """Load csv files for a lexical network into pandas dataframes."""
    if sources is None:
        sources = DEFAULT_DATA_SOURCES.copy()
    if columns is None:
        columns = {k: v.copy() for k, v in DEFAULT_COLUMNS.items()}

    print(f'\x1b[0;34mImporting data from {path}\x1b[0m')

    csv_sources = [src for (src, file) in sources.items() if file.endswith('.csv')]
    ln = {
        src: load_csv(file=sources[src], path=path, columns=columns[src], separator=separator, encoding=encoding)
        for src in csv_sources
    }

    ln['nodes'].set_index('node_id', inplace=True)
    ln['entries'].set_index('entry_id', inplace=True)
    ln['forms'].set_index('node_id', inplace=True)
    ln['labels'].set_index('node_id', inplace=True)
    ln['definitions'].set_index('node_id', inplace=True)
    ln['examples'].set_index('ex_id', inplace=True)

    ln['nodes']['std_name'] = ln['nodes'].apply(lambda row: std_name(row.lexname), axis=1)

    ln['features'].features = ln['features'].features.apply(to_list)
    ln['features'].set_index('node_id', inplace=True)

    ln['propforms'].propform = ln['propforms'].propform.apply(normalize_propforms)
    ln['propforms'].set_index('node_id', inplace=True)

    ln['feature_names'] = load_feature_names(file=sources['feature_names'], path=path, encoding=encoding)
    ln['feature_names'].set_index('feature_id', inplace=True)
    ln['lf_names'] = load_lf_names(file=sources['lf_names'], path=path, encoding=encoding)
    ln['lf_names'].set_index('lf_id', inplace=True)

    return ln
