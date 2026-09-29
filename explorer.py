"""A lightweight, read-only browser explorer for LexNet data."""

from __future__ import annotations

import argparse
import html
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd

import lexnet


EMPTY_WORD_RESULTS = [
    'entry_id', 'entry_name', 'node_id', 'std_name', 'pos', 'match_source'
]
EMPTY_LF_RESULTS = [
    'lf_name', 'source_id', 'source_name', 'target_id', 'target_name',
    'form', 'frame', 'constraint'
]
EMPTY_FEATURE_RESULTS = [
    'entry_id', 'entry_name', 'node_id', 'std_name', 'matching_features'
]


def split_query(text: str | Iterable[str]) -> list[str]:
    """Split comma/newline-separated input while preserving item order."""
    if isinstance(text, str):
        items = re.split(r'[,\n]+', text)
    else:
        items = list(text)
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def _text(value) -> str:
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return ''
    return str(value)


def _matches(series: pd.Series, query: str, mode: str = 'contains') -> pd.Series:
    values = series.fillna('').astype(str).str.casefold()
    query = query.casefold()
    if mode == 'exact':
        return values.eq(query)
    if mode == 'starts with':
        return values.str.startswith(query)
    return values.str.contains(re.escape(query), regex=True)


def _rows_for_index(frame: pd.DataFrame, key) -> pd.DataFrame:
    """Return zero or more rows for an index value as a DataFrame."""
    if frame is None or frame.empty or key not in frame.index:
        return pd.DataFrame(columns=[] if frame is None else frame.columns)
    rows = frame.loc[[key]]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    return rows


class LexnetQueries:
    """Search operations used by the GUI, kept independent of the web UI."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self.data = data
        self.nodes = data['nodes']
        self.entries = data['entries']
        self.node_names = self.nodes['std_name'].fillna('').astype(str).to_dict()
        self.entry_names = self.entries['entry_name'].fillna('').astype(str).to_dict()
        self.node_entries = self.nodes['entry_id'].to_dict()
        self.lf_table = data['lf_names']
        self.lf_names = self.lf_table['lf_name'].fillna('').astype(str).to_dict()
        self.feature_names = data['feature_names']['name'].fillna('').astype(str).to_dict()
        self.form_names = self._name_map(data.get('form_names'))
        self.label_names = self._name_map(data.get('label_names'))

    @staticmethod
    def _name_map(frame: pd.DataFrame | None) -> dict:
        if frame is None or frame.empty or 'name' not in frame:
            return {}
        return frame['name'].fillna('').astype(str).to_dict()

    def search_words(
        self, query: str, mode: str = 'contains', include_forms: bool = True
    ) -> pd.DataFrame:
        """Find lexical entries through entry, unit, or inflected-form text."""
        query = query.strip()
        if not query:
            return pd.DataFrame(columns=EMPTY_WORD_RESULTS)

        entry_ids = set(self.entries.index[_matches(self.entries['entry_name'], query, mode)])
        node_ids = set(self.nodes.index[_matches(self.nodes['std_name'], query, mode)])
        sources: dict[object, set[str]] = {node_id: {'unit'} for node_id in node_ids}

        for node_id, entry_id in self.node_entries.items():
            if entry_id in entry_ids:
                node_ids.add(node_id)
                sources.setdefault(node_id, set()).add('entry')

        forms = self.data.get('forms')
        if include_forms and forms is not None and not forms.empty and 'signifier' in forms:
            matching_forms = forms[_matches(forms['signifier'], query, mode)]
            for node_id in matching_forms.index.unique():
                if node_id in self.nodes.index:
                    node_ids.add(node_id)
                    sources.setdefault(node_id, set()).add('form')

        rows = []
        for node_id in node_ids:
            node = self.nodes.loc[node_id]
            entry_id = node['entry_id']
            feature_rows = _rows_for_index(self.data.get('features'), node_id)
            pos = ''
            if not feature_rows.empty and 'POS' in feature_rows:
                pos_values = (
                    self.feature_names.get(value, _text(value))
                    for value in feature_rows['POS']
                )
                pos = ', '.join(dict.fromkeys(filter(None, pos_values)))
            rows.append({
                'entry_id': entry_id,
                'entry_name': self.entry_names.get(entry_id, ''),
                'node_id': node_id,
                'std_name': _text(node.get('std_name')),
                'pos': pos,
                'match_source': ', '.join(sorted(sources.get(node_id, {'unit'}))),
            })

        return pd.DataFrame(rows, columns=EMPTY_WORD_RESULTS).sort_values(
            ['entry_name', 'std_name'], key=lambda col: col.astype(str).str.casefold()
        ).reset_index(drop=True)

    def _resolve_names(self, queries: Iterable[str], names: dict) -> set:
        resolved = set()
        folded = {key: value.casefold() for key, value in names.items()}
        for query in queries:
            needle = query.casefold()
            exact = {key for key, value in folded.items() if value == needle}
            resolved.update(exact or {key for key, value in folded.items() if needle in value})
        return resolved

    def search_lexical_functions(self, names: str | Iterable[str]) -> pd.DataFrame:
        """Return source/target occurrences for one or more LF names."""
        lf_ids = self._resolve_names(split_query(names), self.lf_names)
        return self.search_lexical_function_ids(lf_ids)

    def search_lexical_function_ids(self, lf_ids: Iterable) -> pd.DataFrame:
        """Return source/target occurrences for LF IDs."""
        lf_ids = set(lf_ids)
        relations = self.data.get('lfs')
        if not lf_ids or relations is None or relations.empty:
            return pd.DataFrame(columns=EMPTY_LF_RESULTS)

        rows = []
        for _, relation in relations[relations['lf_id'].isin(lf_ids)].iterrows():
            source_id = relation['source_id']
            target_id = relation['target_id']
            rows.append({
                'lf_name': self.lf_names.get(relation['lf_id'], _text(relation['lf_id'])),
                'source_id': source_id,
                'source_name': self.node_names.get(source_id, _text(source_id)),
                'target_id': target_id,
                'target_name': self.node_names.get(target_id, _text(target_id)),
                'form': _text(relation.get('form')),
                'frame': _text(relation.get('frame')),
                'constraint': _text(relation.get('constraint')),
            })
        return pd.DataFrame(rows, columns=EMPTY_LF_RESULTS).sort_values(
            ['lf_name', 'source_name', 'target_name'],
            key=lambda col: col.astype(str).str.casefold(),
        ).reset_index(drop=True)

    def lexical_function_ids_for_family(self, family_id) -> list:
        """Return LF IDs belonging to a family, preserving XML order."""
        if 'family_id' not in self.lf_table:
            return []
        return self.lf_table.index[self.lf_table['family_id'] == family_id].tolist()

    def lexical_function_ids_for_group(self, group_index: int) -> list:
        """Return LF IDs belonging to a group, preserving XML order."""
        if 'group_index' not in self.lf_table:
            return []
        return self.lf_table.index[self.lf_table['group_index'] == group_index].tolist()

    def lexical_function_hierarchy(self) -> list[dict]:
        """Return groups containing families containing lexical functions."""
        required = {'group_index', 'family_id', 'family_name'}
        if not required.issubset(self.lf_table.columns):
            return []
        groups = []
        for group_index, group_rows in self.lf_table.groupby('group_index', sort=True):
            families = []
            for family_id, family_rows in group_rows.groupby('family_id', sort=False):
                families.append({
                    'id': _text(family_id),
                    'name': _text(family_rows.iloc[0]['family_name']),
                    'functions': [
                        {'id': _text(lf_id), 'name': _text(row['lf_name'])}
                        for lf_id, row in family_rows.iterrows()
                    ],
                })
            groups.append({'index': int(group_index), 'families': families})
        return groups

    def search_features(
        self, names: str | Iterable[str], require_all: bool = False
    ) -> pd.DataFrame:
        """Return units and entries carrying named grammatical features."""
        feature_ids = self._resolve_names(split_query(names), self.feature_names)
        features = self.data.get('features')
        if not feature_ids or features is None or features.empty:
            return pd.DataFrame(columns=EMPTY_FEATURE_RESULTS)

        by_node: dict[object, set] = {}
        for node_id, row in features.iterrows():
            values = row.get('features', [])
            if not isinstance(values, (list, tuple, set)):
                values = []
            by_node.setdefault(node_id, set()).update(values)

        rows = []
        for node_id, values in by_node.items():
            matching = feature_ids.intersection(values)
            qualifies = feature_ids.issubset(values) if require_all else bool(matching)
            if not qualifies or node_id not in self.nodes.index:
                continue
            node = self.nodes.loc[node_id]
            entry_id = node['entry_id']
            shown_ids = feature_ids if require_all else matching
            rows.append({
                'entry_id': entry_id,
                'entry_name': self.entry_names.get(entry_id, ''),
                'node_id': node_id,
                'std_name': _text(node.get('std_name')),
                'matching_features': ', '.join(
                    sorted((self.feature_names.get(fid, _text(fid)) for fid in shown_ids), key=str.casefold)
                ),
            })
        return pd.DataFrame(rows, columns=EMPTY_FEATURE_RESULTS).sort_values(
            ['entry_name', 'std_name'], key=lambda col: col.astype(str).str.casefold()
        ).reset_index(drop=True)

    def describe_node(self, node_id) -> str:
        """Build a readable, plain-text summary of a lexical unit."""
        if node_id not in self.nodes.index:
            return f'Unknown lexical unit: {node_id}'
        node = self.nodes.loc[node_id]
        entry_id = node['entry_id']
        lines = [
            _text(node.get('std_name')) or _text(node_id),
            f'Entry: {self.entry_names.get(entry_id, _text(entry_id))}',
        ]

        feature_rows = _rows_for_index(self.data.get('features'), node_id)
        if not feature_rows.empty:
            lines.extend(['', 'GRAMMATICAL INFORMATION'])
            for _, row in feature_rows.iterrows():
                parts = []
                for column, label in [('POS', 'POS'), ('ph_str', 'Phraseology'), ('usage', 'Usage')]:
                    value = _text(row.get(column))
                    if value:
                        value = self.feature_names.get(row.get(column), value)
                        parts.append(f'{label}: {value}')
                feature_ids = row.get('features', [])
                if isinstance(feature_ids, (list, tuple, set)):
                    names = [self.feature_names.get(fid, _text(fid)) for fid in feature_ids]
                    if names:
                        parts.append('Features: ' + ', '.join(names))
                if parts:
                    lines.append(' • ' + '; '.join(parts))

        definitions = _rows_for_index(self.data.get('definitions'), node_id)
        if not definitions.empty:
            lines.extend(['', 'DEFINITION'])
            for value in definitions.get('def_HTML', pd.Series(dtype=str)):
                plain = re.sub(r'<[^>]+>', ' ', html.unescape(_text(value)))
                plain = re.sub(r'\s+', ' ', plain).strip()
                if plain:
                    lines.append(plain)

        self._append_wordforms(lines, node_id)
        self._append_semantic_labels(lines, node_id)
        self._append_simple_rows(lines, 'PROPOSITIONAL FORMS', self.data.get('propforms'), node_id, ['propform'])

        relation_links = self.lexical_function_links(node_id)
        if relation_links:
            lines.extend(['', 'LEXICAL FUNCTIONS'])
            lines.extend(link['line'] for link in relation_links)

        ex_rel = self.data.get('ex-rel')
        examples = self.data.get('examples')
        if ex_rel is not None and examples is not None and not ex_rel.empty:
            ex_ids = ex_rel.loc[ex_rel['node_id'] == node_id, 'ex_id']
            matching_examples = examples.loc[examples.index.intersection(ex_ids)]
            if not matching_examples.empty:
                lines.extend(['', 'EXAMPLES'])
                for _, row in matching_examples.iterrows():
                    content = re.sub(r'<[^>]+>', ' ', html.unescape(_text(row.get('content'))))
                    lines.append(' • ' + re.sub(r'\s+', ' ', content).strip())
        return '\n'.join(lines)

    def describe_entry(self, entry_id) -> str:
        """Build a summary of an entry and its lexical units."""
        if entry_id not in self.entries.index:
            return f'Unknown lexical entry: {entry_id}'
        entry = self.entries.loc[entry_id]
        lines = [
            _text(entry.get('entry_name')) or _text(entry_id),
        ]
        metadata = []
        for column, label in [
            ('addtoname', 'Additional name'), ('subscript', 'Subscript'),
            ('superscript', 'Superscript'), ('entry_status', 'Status'),
            ('entry_%', 'Confidence'),
        ]:
            value = _text(entry.get(column))
            if value:
                metadata.append(f'{label}: {value}')
        if metadata:
            lines.extend(['', 'ENTRY INFORMATION', *(' • ' + value for value in metadata)])

        unit_links = self.entry_unit_links(entry_id)
        if unit_links:
            lines.extend(['', 'LEXICAL UNITS'])
            lines.extend(link['line'] for link in unit_links)
        return '\n'.join(lines)

    def entry_unit_links(self, entry_id) -> list[dict[str, str]]:
        """Return the units in an entry with compact grammatical summaries."""
        units = self.nodes[self.nodes['entry_id'] == entry_id]
        links = []
        for node_id, node in units.sort_values(
            'std_name', key=lambda values: values.fillna('').astype(str).str.casefold()
        ).iterrows():
            name = _text(node.get('std_name')) or _text(node_id)
            grammar = self._node_grammar_summary(node_id)
            suffix = f'  [{grammar}]' if grammar else ''
            links.append({
                'line': ' • ' + name + suffix,
                'prefix': ' • ', 'suffix': suffix,
                'item_type': 'node', 'item_id': _text(node_id), 'item_name': name,
            })
        return links

    def _node_grammar_summary(self, node_id) -> str:
        rows = _rows_for_index(self.data.get('features'), node_id)
        if rows.empty:
            return ''
        values = []
        for _, row in rows.iterrows():
            pos = row.get('POS')
            if _text(pos):
                values.append(self.feature_names.get(pos, _text(pos)))
            feature_ids = row.get('features', [])
            if isinstance(feature_ids, (list, tuple, set)):
                values.extend(self.feature_names.get(fid, _text(fid)) for fid in feature_ids)
        return '; '.join(dict.fromkeys(filter(None, values)))

    def inspector_payload(self, item_type: str, item_id) -> dict:
        """Return a generic payload for the browser inspector."""
        if item_type == 'node':
            entry_id = self.node_entries.get(item_id)
            links = []
            if entry_id is not None:
                name = self.entry_names.get(entry_id, _text(entry_id))
                prefix, suffix = 'Entry: ', ''
                links.append({
                    'line': prefix + name + suffix,
                    'prefix': prefix, 'suffix': suffix,
                    'item_type': 'entry', 'item_id': _text(entry_id), 'item_name': name,
                })
            links.extend(self.lexical_function_links(item_id))
            return {
                'title': 'Lexical unit', 'description': self.describe_node(item_id),
                'links': links,
            }
        if item_type == 'entry':
            return {
                'title': 'Lexical entry', 'description': self.describe_entry(item_id),
                'links': self.entry_unit_links(item_id),
            }
        return {'title': 'Inspector', 'error': f'Unknown item type: {item_type}'}

    def lexical_function_links(self, node_id) -> list[dict[str, str]]:
        """Return display text and navigation targets for a unit's LF relations."""
        relations = self.data.get('lfs')
        if relations is None or relations.empty:
            return []
        links = []
        outgoing = relations[relations['source_id'] == node_id]
        incoming = relations[relations['target_id'] == node_id]
        for _, row in outgoing.iterrows():
            name = self.node_names.get(row['target_id'], _text(row['target_id']))
            prefix = f" → {self.lf_names.get(row['lf_id'], row['lf_id'])}: "
            links.append({
                'line': prefix + name, 'prefix': prefix, 'suffix': '',
                'item_type': 'node', 'item_id': _text(row['target_id']), 'item_name': name,
            })
        for _, row in incoming.iterrows():
            name = self.node_names.get(row['source_id'], _text(row['source_id']))
            prefix = f" ← {self.lf_names.get(row['lf_id'], row['lf_id'])}: "
            links.append({
                'line': prefix + name, 'prefix': prefix, 'suffix': '',
                'item_type': 'node', 'item_id': _text(row['source_id']), 'item_name': name,
            })
        return links

    def _append_wordforms(self, lines, node_id):
        rows = _rows_for_index(self.data.get('forms'), node_id)
        rendered = []
        for _, row in rows.iterrows():
            signifier = _text(row.get('signifier'))
            feature_ids = split_query(_text(row.get('features')).strip('()'))
            features = ', '.join(filter(None, (
                self.form_names.get(fid, '' if fid.startswith('ls:') else fid)
                for fid in feature_ids
            )))
            value = ' — '.join(filter(None, [signifier, features]))
            if value:
                rendered.append(' • ' + value)
        if rendered:
            lines.extend(['', 'WORDFORMS', *rendered])

    def _append_semantic_labels(self, lines, node_id):
        rows = _rows_for_index(self.data.get('labels'), node_id)
        rendered = []
        for _, row in rows.iterrows():
            label_id = row.get('label')
            name = self.label_names.get(label_id, 'Unknown semantic label')
            if name:
                rendered.append(' • ' + name)
        if rendered:
            lines.extend(['', 'SEMANTIC LABELS', *rendered])

    @staticmethod
    def _append_simple_rows(lines, title, frame, node_id, columns):
        rows = _rows_for_index(frame, node_id)
        if rows.empty:
            return
        rendered = []
        for _, row in rows.iterrows():
            values = [_text(row.get(column)) for column in columns]
            value = ' — '.join(filter(None, values))
            if value:
                rendered.append(' • ' + value)
        if rendered:
            lines.extend(['', title, *rendered])


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LexNet Explorer</title>
<style>
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #17212b; background: #f3f5f7; }
* { box-sizing: border-box; }
body { margin: 0; }
header { padding: 16px 24px; background: #17324d; color: white; display: flex; align-items: baseline; gap: 16px; }
h1 { margin: 0; font-size: 21px; }
header span { opacity: .75; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
main { height: calc(100vh - 58px); padding: 16px; display: grid; grid-template-columns: minmax(520px, 3fr) minmax(340px, 2fr); gap: 16px; }
.card { background: white; border: 1px solid #d8dee5; border-radius: 8px; overflow: hidden; min-height: 0; }
.left { display: flex; flex-direction: column; }
.tabs { display: flex; border-bottom: 1px solid #d8dee5; }
.tab { border: 0; background: transparent; padding: 13px 17px; cursor: pointer; font-weight: 600; color: #536475; }
.tab.active { color: #0866a8; box-shadow: inset 0 -3px #1683c7; }
.panel { display: none; padding: 14px; border-bottom: 1px solid #d8dee5; }
.panel.active { display: block; }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
input[type=text] { flex: 1; min-width: 210px; padding: 8px 10px; border: 1px solid #aeb9c4; border-radius: 5px; font: inherit; }
select, button.search { padding: 8px 11px; border: 1px solid #97a6b4; border-radius: 5px; background: white; font: inherit; }
button.search { background: #0878ba; border-color: #0878ba; color: white; cursor: pointer; font-weight: 600; }
.tree-select { position: relative; min-width: 220px; }
.tree-select-toggle { width: 100%; padding: 8px 28px 8px 11px; border: 1px solid #97a6b4; border-radius: 5px; background: white; color: #17212b; font: inherit; text-align: left; cursor: pointer; position: relative; }
.tree-select-toggle::after { content: '▾'; position: absolute; right: 9px; color: #536475; }
.tree-select-menu { position: absolute; z-index: 20; top: calc(100% + 3px); left: 0; min-width: 100%; max-height: 360px; overflow-y: auto; padding: 4px 0; background: white; border: 1px solid #aeb9c4; border-radius: 5px; box-shadow: 0 5px 16px #17212b24; }
.tree-select-menu[hidden] { display: none; }
.tree-select-item { display: block; width: 100%; padding: 6px 12px; border: 0; background: white; color: #17212b; font: inherit; text-align: left; white-space: nowrap; cursor: pointer; }
.tree-select-item:hover, .tree-select-item:focus { background: #edf6fc; outline: none; }
.tree-select-group { color: #6d7882; font-weight: 600; background: #f3f5f7; }
.tree-select-family { padding-left: 28px; }
.options { margin-top: 9px; display: flex; gap: 18px; font-size: 14px; }
.status { padding: 8px 14px; color: #536475; background: #f7f9fa; border-bottom: 1px solid #d8dee5; font-size: 13px; }
.table-wrap { overflow: auto; flex: 1; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { position: sticky; top: 0; background: #f0f3f6; text-align: left; padding: 8px 10px; border-bottom: 1px solid #cbd3db; cursor: pointer; user-select: none; }
th:hover, th:focus { background: #e3e9ee; outline: none; }
td { padding: 8px 10px; border-bottom: 1px solid #e7ebef; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
tbody tr:hover { background: #edf6fc; }
.node-link { padding: 0; border: 0; background: transparent; color: #0878ba; font: inherit; text-decoration: underline; cursor: pointer; }
.node-link:hover, .node-link:focus { color: #064f79; }
.details { display: flex; flex-direction: column; }
.details h2 { font-size: 16px; margin: 0; padding: 14px; border-bottom: 1px solid #d8dee5; }
pre { flex: 1; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; padding: 16px; font: 14px/1.5 system-ui, sans-serif; }
@media (max-width: 850px) { main { height: auto; grid-template-columns: 1fr; } .card { min-height: 500px; } .details { min-height: 500px; } }
</style>
</head>
<body>
<header><h1>LexNet Explorer</h1><span id="dataset"></span></header>
<main>
  <section class="card left">
    <nav class="tabs">
      <button class="tab active" data-kind="word">Words</button>
      <button class="tab" data-kind="lf">Lexical functions</button>
      <button class="tab" data-kind="feature">Features</button>
    </nav>
    <form class="panel active" id="word-panel" data-kind="word">
      <div class="controls"><input type="text" name="q" placeholder="Search a word…" autofocus><select name="mode"><option>contains</option><option>starts with</option><option>exact</option></select><button class="search">Search</button></div>
      <label class="options"><span><input type="checkbox" name="forms" checked> Include inflected forms</span></label>
    </form>
    <form class="panel" id="lf-panel" data-kind="lf">
      <div class="controls">
        <div class="tree-select" id="lf-family">
          <button type="button" class="tree-select-toggle" aria-haspopup="listbox" aria-expanded="false">Choose a family…</button>
          <div class="tree-select-menu" role="listbox" hidden></div>
          <input type="hidden" name="family_id" value="">
        </div>
        <select name="lf_id" id="lf-function" disabled><option value="">Choose a family first</option></select>
      </div>
      <div class="controls" style="margin-top: 8px"><input type="text" name="q" list="lf-names" placeholder="Or enter LF names: Magn, Oper1, …"><datalist id="lf-names"></datalist><button class="search">Search</button></div>
    </form>
    <form class="panel" id="feature-panel" data-kind="feature">
      <div class="controls"><input type="text" name="q" list="feature-names" placeholder="locution forte, …"><datalist id="feature-names"></datalist><button class="search">Search</button></div>
      <div class="options"><label><input type="radio" name="combine" value="any" checked> Match any</label><label><input type="radio" name="combine" value="all"> Match all</label></div>
    </form>
    <div class="status" id="status">Ready.</div>
    <div class="table-wrap"><table><thead><tr id="head"></tr></thead><tbody id="results"></tbody></table></div>
  </section>
  <section class="card details"><h2 id="inspector-title">Inspector</h2><pre id="details">Select a result to inspect it.</pre></section>
</main>
<script>
const columns = {
  word: [['entry_name','Entry'], ['std_name','Lexical unit'], ['pos','POS']],
  lf: [['lf_name','Function'], ['source_name','Source'], ['target_name','Target'], ['form','Form']],
  feature: [['entry_name','Entry'], ['std_name','Lexical unit'], ['matching_features','Matching features']]
};
let kind = 'word';
let currentRows = [];
let sortKey = null;
let sortAscending = true;
const collator = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const head = document.querySelector('#head');

function selectTab(next) {
  kind = next;
  currentRows = []; sortKey = null; sortAscending = true;
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.dataset.kind === kind));
  document.querySelectorAll('.panel').forEach(x => x.classList.toggle('active', x.dataset.kind === kind));
  renderHead(); results.replaceChildren(); status.textContent = 'Ready.';
  document.querySelector(`#${kind}-panel input[name=q]`).focus();
}
function renderHead() {
  head.replaceChildren(...columns[kind].map(([key, label]) => {
    const th=document.createElement('th');
    th.tabIndex=0;
    th.textContent=label + (sortKey === key ? (sortAscending ? ' ▲' : ' ▼') : '');
    th.setAttribute('aria-sort', sortKey === key ? (sortAscending ? 'ascending' : 'descending') : 'none');
    th.addEventListener('click', () => sortBy(key));
    th.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); sortBy(key); } });
    return th;
  }));
}
function sortBy(key) {
  if (sortKey === key) sortAscending = !sortAscending;
  else { sortKey = key; sortAscending = true; }
  currentRows.sort((left, right) => {
    const a = String(left[key] || ''); const b = String(right[key] || '');
    if (!a && b) return 1; if (a && !b) return -1;
    return collator.compare(a, b) * (sortAscending ? 1 : -1);
  });
  renderHead(); renderRows();
}
function renderRows() {
  const fragment = document.createDocumentFragment();
  for (const row of currentRows) {
    const tr = document.createElement('tr');
    for (const [key] of columns[kind]) {
      const td=document.createElement('td'); td.title=row[key] || '';
      if ((kind === 'word' || kind === 'feature') && (key === 'entry_name' || key === 'std_name')) {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link'; link.textContent=row[key] || '';
        const itemType = key === 'entry_name' ? 'entry' : 'node';
        const itemId = key === 'entry_name' ? row.entry_id : row.node_id;
        link.addEventListener('click', () => showItem(itemType, itemId));
        td.append(link);
      } else if (kind === 'lf' && (key === 'source_name' || key === 'target_name' || (key === 'form' && row.form))) {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link'; link.textContent=row[key] || '';
        const nodeId = key === 'source_name' ? row.source_id : row.target_id;
        link.addEventListener('click', event => { event.stopPropagation(); showItem('node', nodeId); });
        td.append(link);
      } else td.textContent=row[key] || '';
      tr.append(td);
    }
    fragment.append(tr);
  }
  results.replaceChildren(fragment);
}
async function search(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const params = new URLSearchParams(new FormData(form));
  params.set('kind', form.dataset.kind);
  if (form.dataset.kind === 'word') params.set('forms', form.elements.forms.checked ? '1' : '0');
  status.textContent = 'Searching…'; results.replaceChildren();
  try {
    const response = await fetch('/api/search?' + params);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Search failed');
    status.textContent = payload.status;
    currentRows = payload.rows; sortKey = null; sortAscending = true;
    renderHead(); renderRows();
  } catch (error) { status.textContent = error.message; }
}
async function showItem(itemType, itemId) {
  const params = new URLSearchParams({type: itemType, id: itemId});
  const response = await fetch('/api/inspect?' + params);
  const payload = await response.json();
  renderDetails(payload);
}
function renderDetails(payload) {
  const details = document.querySelector('#details');
  document.querySelector('#inspector-title').textContent = payload.title || 'Inspector';
  if (!payload.description) { details.textContent = payload.error || 'Unable to load item.'; return; }
  const links = new Map();
  for (const link of payload.links || []) {
    if (!links.has(link.line)) links.set(link.line, []);
    links.get(link.line).push(link);
  }
  const content = document.createDocumentFragment();
  const lines = payload.description.split('\n');
  lines.forEach((line, index) => {
    const choices = links.get(line);
    const relation = choices && choices.shift();
    if (relation) {
      content.append(document.createTextNode(relation.prefix));
      const link = document.createElement('button');
      link.type='button'; link.className='node-link'; link.textContent=relation.item_name;
      link.addEventListener('click', () => showItem(relation.item_type, relation.item_id));
      content.append(link);
      content.append(document.createTextNode(relation.suffix || ''));
    } else content.append(document.createTextNode(line));
    if (index < lines.length - 1) content.append(document.createTextNode('\n'));
  });
  details.replaceChildren(content);
}
async function initialize() {
  const response = await fetch('/api/meta'); const meta = await response.json();
  document.querySelector('#dataset').textContent = `${meta.path} — ${meta.entries.toLocaleString()} entries, ${meta.units.toLocaleString()} units`;
  for (const [target, values] of [['lf-names', meta.lexical_functions], ['feature-names', meta.features]]) {
    const list=document.querySelector('#'+target); for (const value of values) { const option=document.createElement('option'); option.value=value; list.append(option); }
  }
  const familyControl = document.querySelector('#lf-family');
  const familyToggle = familyControl.querySelector('.tree-select-toggle');
  const familyMenu = familyControl.querySelector('.tree-select-menu');
  const familyValue = familyControl.querySelector('input[name=family_id]');
  const functionSelect = document.querySelector('#lf-function');
  const nameInput = document.querySelector('#lf-panel input[name=q]');
  function chooseFamily(value, label, clearName=true) {
    familyValue.value=value; familyToggle.textContent=label;
    familyMenu.hidden=true; familyToggle.setAttribute('aria-expanded', 'false');
    if (clearName) nameInput.value='';
    if (value.startsWith('group:')) {
      const groupNumber=value.slice(6);
      functionSelect.replaceChildren();
      const all=document.createElement('option'); all.value=''; all.textContent=`All functions in Group ${groupNumber}`; functionSelect.append(all);
      functionSelect.disabled=true;
      return;
    }
    const family = meta.lf_hierarchy.flatMap(group => group.families).find(item => item.id === value);
    functionSelect.replaceChildren();
    const all=document.createElement('option'); all.value=''; all.textContent=family ? 'All functions in family' : 'Choose a family first'; functionSelect.append(all);
    for (const lf of family?.functions || []) { const option=document.createElement('option'); option.value=lf.id; option.textContent=lf.name; functionSelect.append(option); }
    functionSelect.disabled = !family;
  }
  for (const group of meta.lf_hierarchy) {
    const groupItem=document.createElement('button'); groupItem.type='button'; groupItem.className='tree-select-item tree-select-group'; groupItem.setAttribute('role', 'option'); groupItem.textContent=`Group ${group.index}`;
    groupItem.addEventListener('click', () => chooseFamily(`group:${group.index}`, `Group ${group.index}`)); familyMenu.append(groupItem);
    for (const family of group.families) {
      const familyItem=document.createElement('button'); familyItem.type='button'; familyItem.className='tree-select-item tree-select-family'; familyItem.setAttribute('role', 'option'); familyItem.textContent=family.name;
      familyItem.addEventListener('click', () => chooseFamily(family.id, family.name)); familyMenu.append(familyItem);
    }
  }
  familyToggle.addEventListener('click', () => {
    familyMenu.hidden=!familyMenu.hidden;
    familyToggle.setAttribute('aria-expanded', String(!familyMenu.hidden));
  });
  document.addEventListener('click', event => { if (!familyControl.contains(event.target)) { familyMenu.hidden=true; familyToggle.setAttribute('aria-expanded', 'false'); } });
  functionSelect.addEventListener('change', () => { nameInput.value = ''; });
  nameInput.addEventListener('input', () => {
    if (!nameInput.value) return;
    chooseFamily('', 'Choose a family…', false);
  });
}
document.querySelectorAll('.tab').forEach(x => x.addEventListener('click', () => selectTab(x.dataset.kind)));
document.querySelectorAll('form').forEach(x => x.addEventListener('submit', search));
renderHead(); initialize();
</script>
</body>
</html>'''


def _records(frame: pd.DataFrame, limit: int = 2000) -> list[dict[str, str]]:
    return [
        {column: _text(value) for column, value in row.items()}
        for row in frame.head(limit).to_dict(orient='records')
    ]


def create_server(
    queries: LexnetQueries, data_path: str, host: str = '127.0.0.1', port: int = 0
) -> ThreadingHTTPServer:
    """Create the local HTTP server without starting it."""
    class ExplorerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            params = parse_qs(request.query)
            try:
                if request.path == '/':
                    return self._send(PAGE, 'text/html; charset=utf-8')
                if request.path == '/favicon.ico':
                    return self._send(b'', 'image/x-icon', status=204)
                if request.path == '/api/meta':
                    return self._json({
                        'path': data_path,
                        'entries': len(queries.entries),
                        'units': len(queries.nodes),
                        'lexical_functions': sorted(set(queries.lf_names.values()), key=str.casefold),
                        'lf_hierarchy': queries.lexical_function_hierarchy(),
                        'features': sorted(set(queries.feature_names.values()), key=str.casefold),
                    })
                if request.path == '/api/inspect':
                    return self._json(queries.inspector_payload(
                        self._param(params, 'type'), self._param(params, 'id')
                    ))
                if request.path == '/api/node':
                    return self._json(queries.inspector_payload('node', self._param(params, 'id')))
                if request.path == '/api/search':
                    return self._search(params)
                return self._json({'error': 'Not found'}, status=404)
            except Exception as error:
                return self._json({'error': str(error)}, status=500)

        def _search(self, params):
            kind = self._param(params, 'kind')
            query = self._param(params, 'q')
            if kind == 'word':
                result = queries.search_words(
                    query,
                    mode=self._param(params, 'mode', 'contains'),
                    include_forms=self._param(params, 'forms') == '1',
                )
                count = result.entry_id.nunique()
                summary = f'{len(result):,} lexical units in {count:,} entries.'
            elif kind == 'lf':
                lf_id = self._param(params, 'lf_id')
                family_id = self._param(params, 'family_id')
                if query:
                    result = queries.search_lexical_functions(query)
                elif lf_id:
                    result = queries.search_lexical_function_ids([lf_id])
                elif family_id.startswith('group:'):
                    result = queries.search_lexical_function_ids(
                        queries.lexical_function_ids_for_group(int(family_id.removeprefix('group:')))
                    )
                else:
                    result = queries.search_lexical_function_ids(
                        queries.lexical_function_ids_for_family(family_id)
                    )
                summary = f'{len(result):,} lexical-function relations.'
            elif kind == 'feature':
                result = queries.search_features(
                    query, require_all=self._param(params, 'combine') == 'all'
                )
                count = result.entry_id.nunique()
                summary = f'{len(result):,} lexical units in {count:,} entries.'
            else:
                return self._json({'error': 'Unknown search type'}, status=400)
            if len(result) > 2000:
                summary += ' Showing the first 2,000 rows.'
            return self._json({'status': summary, 'rows': _records(result)})

        @staticmethod
        def _param(params, name, default=''):
            return params.get(name, [default])[0]

        def _json(self, payload, status=200):
            return self._send(
                json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                'application/json; charset=utf-8', status,
            )

        def _send(self, body, content_type, status=200):
            if isinstance(body, str):
                body = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return ThreadingHTTPServer((host, port), ExplorerHandler)


def launch(
    data_path: str, host: str = '127.0.0.1', port: int = 0, open_browser: bool = True
) -> None:
    """Load a network and serve its browser interface until interrupted."""
    network = lexnet.LexicalNetwork(data_path)
    server = create_server(LexnetQueries(network.data), data_path, host, port)
    url = f'http://{host}:{server.server_port}/'
    print(f'LexNet Explorer: {url}')
    print('Press Ctrl-C to stop.')
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping LexNet Explorer.')
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Explore a LexNet export in a lightweight GUI.')
    parser.add_argument('data', help='Path to a LexNet data folder')
    parser.add_argument('--port', type=int, default=0, help='Local port (default: choose automatically)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
    args = parser.parse_args()
    launch(args.data, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
