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
columns: source,    target,    type,    subtype
renamed: cp_source, cp_target, cp_type, cp_subtype
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
index  : default


Semantic labels
=======================================
Semantic labels for the nodes.

file   : '10-lssemlabel-rel.csv'
columns: node,    label %
renamed: node_id, label, label_%
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

file   : '13-lslf-rel.csv'
columns: source,    lf, target,    form, separator, merged, syntacticframe, constraint, position
renamed: source_id, lf_id, target_id, form, separator, merged, frame,          constraint, position
index  : default


Definitions
=======================================
Definitions for the nodes.

file   : '17-lsdef.csv'
columns: node,    def_XML, def_HTML
renamed: node_id, def_XML, def_HTML
index  : node_id


TODO: load XML data
    05-lsgramcharac-model.xml (grammatical features)
    09-lssemlabel-model.xml
    12-lslf-model.xml
TODO: should we use IDs as row index? (cf. pos_names in load_pos_feature_names())
"""

import argparse
import pandas as pd
from pandas import *
import xml.etree.ElementTree as ET
import re

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
    'ex-rel': '18-lsex-rel.csv'
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

def load_csv(file, path, columns=DEFAULT_COLUMNS, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
    data = pd.read_csv('/'.join([path, file]), sep=separator, encoding=encoding, header=0, names=columns)
    print_imported(data, file)
    return data

def load_xml(file, path, encoding=DEFAULT_ENCODING):
    # ET.parse() doesn't encoding specification
    with open('/'.join([path, file]), 'r', encoding=encoding) as f:
        xml = ET.fromstring(f.read())
    return xml

def load_feature_names(file, path, encoding=DEFAULT_ENCODING):
    xml = load_xml(file=file, path=path, encoding=encoding)
    features = xml.iter('characteristic')
    feature_names = pd.DataFrame([{'feature_id':f.get('id'), 'name':f.get('name')} for f in features])
    print_imported(feature_names, file, items='grammatical features')
    return feature_names

def load_lf_names(file, path, encoding=DEFAULT_ENCODING):
    xml = load_xml(file=file, path=path, encoding=encoding)
    tags = xml.findall('group/family/lexicalfunction')
    attributes = ['id', 'name']
    cols = ['lf_id', 'lf_name']
    assert len(attributes) == len(cols)
    lf_names = pd.DataFrame([{'lf_id':tag.get('id'), 'lf_name':tag.get('name'), 'type':tag.get('linktype')} for tag in tags])
    print_imported(lf_names, file)
    return lf_names

def std_name(lexname):
    """Returns the standard name for a lexical unit"""
    lexname = re.sub(r"<span class='namingform'>([^<]+)</span>", r'\1', lexname)
    lexname = re.sub(r"<span class='(sup|sub)'>([^<]+)</span>", r'_\2', lexname)
    lexname = re.sub(r"<span class='num'>([^<]+)</span>", r'#\1', lexname)
    return lexname

def normalize_propforms(propform):
    """
    Fix some common bugs in propositional forms.
    Propositional forms starting with a "!" need to be validated.
    These "!" are removed and the propositional form is considered valid.  # FIXME disabled
    """
    propform = re.sub(r"'", r'’', propform)
    propform = re.sub(r' +,', r',', propform)
    propform = re.sub(r'(\w)~', r'\1 ~', propform)
    propform = re.sub(r'~(\w)', r'~ \1', propform)
    propform = re.sub(r'\s\s+', r' ', propform)
    # propform = re.sub(r'^! ?', r'', propform)
    return propform.strip()

def to_list(text):
    """Convert a LN list such as '(ls:fr:gc:26,ls:fr:gc:73)' to a python list"""
    if type(text) == str:
        assert text[0] == '('
        assert text[-1] == ')'
        return text[1:-1].split(',')
    else:
        return []

def load(path, sources=DEFAULT_DATA_SOURCES, columns=DEFAULT_COLUMNS, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
    """
    Loads csv files for a lexical network into pandas dataframes.
    Parameters
        path (str):         directory where the csv files reside (same for all files)
        sources (dict):     name of csv file for each table
        columns (dict):     column names for each table
        separator (str):    csv separator (same for all files)
        encoding (str):     character encoding for the files (same for all files)
    Returns
        dict:   table names with their corresponding dataframe
    """

    # Import data
    print(f'\x1b[0;34mImporting data from {path}\x1b[0m')

    # Load CSV files
    csv_sources = [src for (src, file) in sources.items() if file.endswith('.csv')]
    ln = {src:load_csv(file=sources[src], path=path, columns=columns[src], separator=separator, encoding=encoding) for src in csv_sources}

    # Reindex
    ln['nodes'].set_index('node_id', inplace=True)
    ln['entries'].set_index('entry_id', inplace=True)
    ln['forms'].set_index('node_id', inplace=True)
    ln['labels'].set_index('node_id', inplace=True)
    ln['definitions'].set_index('node_id', inplace=True)
    ln['examples'].set_index('ex_id', inplace=True)

    # Add a column to the nodes table for standard names
    ln['nodes']['std_name'] = ln['nodes'].apply(lambda row: std_name(row.lexname), axis=1)

    # Convert grammatical features to list
    ln['features'].features = ln['features'].features.apply(to_list)
    ln['features'].set_index('node_id', inplace=True)

    # Normalize propositional forms
    ln['propforms'].propform = ln['propforms'].propform.apply(normalize_propforms)
    ln['propforms'].set_index('node_id', inplace=True)

    # Load XML files
    ln['feature_names'] = load_feature_names(file=sources['feature_names'], path=path, encoding=encoding)
    ln['feature_names'].set_index('feature_id', inplace=True)
    ln['lf_names'] = load_lf_names(file=sources['lf_names'], path=path, encoding=encoding)
    ln['lf_names'].set_index('lf_id', inplace=True)

    return ln


def test(path, separator, encoding):

    # Load Lexical Network
    ln = load(path=path, separator=separator, encoding=encoding)

    # Show sample
    print(f'\nSample from {len(ln)} tables:\n')
    for table, data in ln.items():
        print(f'\x1b[0;34m{table}\x1b[0m')
        print(data)#.head()
        print()

    return ln

if __name__ == '__main__':

    print(f"\x1b[0;31mRunning in test mode\x1b[0m")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create dataframes from a lexical network.')
    parser.add_argument('-d', '--data', required=True, help='Path to the data folder')
    parser.add_argument('-s', '--separator', default=DEFAULT_SEPARATOR, help='CSV separator')
    parser.add_argument('-e', '--encoding', default=DEFAULT_ENCODING, help='character encoding')
    args = parser.parse_args()

    ln = test(path=args.data, separator=args.separator, encoding=args.encoding)
