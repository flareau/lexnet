"""
Imports the data from a Lexical Network into the following pandas dataframes:


Nodes
=======================================
A node in the network represents a lexical unit.

file   : 01-lsnodes.csv
columns: id,      entry,    lexnum, status,      %,      update_date,      update_time,      lexname
renamed: node_id, entry_id, lexnum, node_status, node_%, node_update_date, node_update_time, lexname
index  : node_id


Entries
=======================================
Nodes are grouped into lexical entries.

file   : 02-lsentries.csv
columns: id,       addtoname, name,       subscript, superscript, status,       %
renamed: entry_id, addtoname, entry_name, subscript, superscript, entry_status, entry_%
index  : entry_id


Copolysemy relations
=======================================
Nodes of an entry are linked by copolysemy relations.

file   : 04-lscopolysemy-rel.csv
columns: source,         target,         type,    subtype
renamed: source_node_id, target_node_id, cp_type, cp_subtype
index  : default


Grammatical features
=======================================
Part of speech, gender, etc., for each node.

file   : '06-lsgramcharac-rel.csv'
columns: node,    usagenote, usagenotevars, POS, phraseolstruc, embeddedlex, othercharac, othercharacvars
renamed: node_id, usage,     usagevars,     POS, ph_str,        embeddedlex, features,    featuresvars
index  : node_id


Wordforms
=======================================
Inflected forms for each node.

file   : '08-lswordforms.csv'
columns: node,    features, signifier
renamed: node_id, features, signifier
index  : node_id


Semantic labels
=======================================
Semantic labels for the nodes.

file   : '10-lssemlabel-rel.csv'
columns: node,    label,            %
renamed: node_id, semantic_label_id, label_%
index  : node_id


Propositional forms
=======================================
Propositional forms for the nodes.

file   : '11-lspropform-rel.csv'
columns: node,    propform, tildevalue, %,               actantslist
renamed: node_id, propform, tildevalue, propform_confid, actants
index  : node_id


Lexical functions
=======================================
Lexical functions for the nodes.

file   : '15-lslf-rel.csv'
columns: source,         lf,                  target,         form, separator, merged, syntacticframe, constraint, position
renamed: source_node_id, lexical_function_id, target_node_id, form, separator, merged, frame,          constraint, position
index  : default


Definitions
=======================================
Definitions for the nodes.

file   : '13-lsdef.csv'
columns: node,    def_XML, def_HTML
renamed: node_id, def_XML, def_HTML
index  : node_id


Feature names
=======================================
Grammatical feature names from XML.

file   : '05-lsgramcharac-model.xml'
columns: id,      name
renamed: feature_id, name
index  : feature_id


LF names
=======================================
Lexical function names from XML.

file   : '14-lslf-model.xml'
columns: id,                  name,    linktype
renamed: lexical_function_id, lf_name, type
index  : lexical_function_id


Examples
=======================================
Examples linked to lexical units.

file   : '17-lsex.csv'
columns: id,         source, status, content, title, authors, location, date
renamed: example_id, source, status, content, title, authors, location, date
index  : example_id


Example relations
=======================================
Link table from lexical units to examples.

file   : '18-lsex-rel.csv'
columns: node,    ex,         occurrence, position, %
renamed: node_id, example_id, occurrence, position, %
index  : default


TODO: should we use IDs as row index? (cf. pos_names in load_pos_feature_names())
"""

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
    'copolysemy': ['source_node_id', 'target_node_id', 'cp_type', 'cp_subtype'],
    'features': ['node_id', 'usage', 'usagevars', 'POS', 'ph_str', 'embeddedlex', 'features', 'featuresvars'],
    'forms': ['node_id', 'features', 'signifier'],
    'labels': ['node_id', 'semantic_label_id', 'label_%'],
    'propforms': ['node_id', 'propform', 'tildevalue', 'propform_confid', 'actants'],
    'lfs': ['source_node_id', 'lexical_function_id', 'target_node_id', 'form', 'separator', 'merged', 'frame', 'constraint', 'position'],
    'definitions': ['node_id', 'def_XML', 'def_HTML'],
    'examples': ['example_id', 'source', 'status', 'content', 'title', 'authors', 'location', 'date'],
    'ex-rel': ['node_id', 'example_id', 'occurrence', 'position', '%'],
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


def load_form_names(file, path, encoding=DEFAULT_ENCODING):
    xml = load_xml(file=file, path=path, encoding=encoding)
    form_names = pd.DataFrame([
        {'wordform_feature_id': tag.get('id'), 'name': tag.get('name')}
        for tag in xml.iter('feature')
    ])
    print_imported(form_names, file, items='wordform features')
    return form_names


def load_label_names(file, path, encoding=DEFAULT_ENCODING):
    xml = load_xml(file=file, path=path, encoding=encoding)
    label_names = pd.DataFrame([
        {'semantic_label_id': tag.get('id'), 'name': tag.get('name')}
        for tag in xml.iter('instance')
    ])
    name_counts = label_names.groupby('semantic_label_id', dropna=False)['name'].nunique(dropna=False)
    conflicting_ids = name_counts[name_counts > 1].index.tolist()
    if conflicting_ids:
        raise ValueError(f'Conflicting names for semantic label IDs: {conflicting_ids}')
    label_names.drop_duplicates('semantic_label_id', inplace=True)
    print_imported(label_names, file, items='semantic labels')
    return label_names


def load_lf_names(file, path, encoding=DEFAULT_ENCODING):
    xml = load_xml(file=file, path=path, encoding=encoding)
    rows = []
    for group_index, group in enumerate(xml.findall('group'), start=1):
        for family_index, family in enumerate(group.findall('family'), start=1):
            for lf_index, tag in enumerate(family.findall('lexicalfunction'), start=1):
                rows.append({
                    'lexical_function_id': tag.get('id'),
                    'lf_name': tag.get('name'),
                    'type': tag.get('linktype'),
                    'family_id': family.get('id'),
                    'family_name': family.get('name'),
                    'group_index': group_index,
                    'family_index': family_index,
                    'lf_index': lf_index,
                })
    lf_names = pd.DataFrame(rows)
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
    if not isinstance(text, str):
        return []
    if len(text) < 2 or not text.startswith('(') or not text.endswith(')'):
        raise ValueError(f'Invalid LexNet list: {text!r}')
    content = text[1:-1]
    return content.split(',') if content else []


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
    ln['entries']['superscript'] = pd.to_numeric(
        ln['entries']['superscript'], errors='raise'
    ).astype('Int64')
    ln['entries'].set_index('entry_id', inplace=True)
    ln['forms'].set_index('node_id', inplace=True)
    ln['labels'].set_index('node_id', inplace=True)
    ln['definitions'].set_index('node_id', inplace=True)
    ln['examples'].set_index('example_id', inplace=True)

    ln['nodes']['std_name'] = ln['nodes'].apply(lambda row: std_name(row.lexname), axis=1)

    ln['features'].features = ln['features'].features.apply(to_list)
    ln['features'].set_index('node_id', inplace=True)

    ln['propforms'].propform = ln['propforms'].propform.apply(normalize_propforms)
    ln['propforms'].set_index('node_id', inplace=True)

    ln['feature_names'] = load_feature_names(file=sources['feature_names'], path=path, encoding=encoding)
    ln['feature_names'].set_index('feature_id', inplace=True)
    ln['form_names'] = load_form_names(file=sources['form_names'], path=path, encoding=encoding)
    ln['form_names'].set_index('wordform_feature_id', inplace=True)
    ln['label_names'] = load_label_names(file=sources['label_names'], path=path, encoding=encoding)
    ln['label_names'].set_index('semantic_label_id', inplace=True)
    ln['lf_names'] = load_lf_names(file=sources['lf_names'], path=path, encoding=encoding)
    ln['lf_names'].set_index('lexical_function_id', inplace=True)

    return ln
