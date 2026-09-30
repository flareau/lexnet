"""Search and inspection operations for LexNet explorer data."""

import html
import re

import pandas as pd


EMPTY_WORD_RESULTS = [
    'entry_id', 'entry_name', 'entry_label', 'node_id', 'std_name', 'unit_label',
    'pos', 'match_source'
]
EMPTY_LF_RESULTS = [
    'lf_name', 'source_node_id', 'source_name', 'target_node_id', 'target_name',
    'source_label', 'target_label', 'form', 'frame', 'constraint'
]
EMPTY_FEATURE_RESULTS = [
    'entry_id', 'entry_name', 'entry_label', 'node_id', 'std_name', 'unit_label',
    'matching_features'
]


def split_query(text):
    """Split comma/newline-separated input while preserving item order."""
    if isinstance(text, str):
        items = re.split(r'[,\n]+', text)
    else:
        items = list(text)
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def _text(value):
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return ''
    return str(value)


def _number(value):
    """Return a sortable number, placing missing or malformed values last."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return float('inf')


def _name_text(label):
    """Return the searchable plain-text form of a structured lexical name."""
    text = label['name']
    if label.get('subscript'):
        text += '_' + label['subscript']
    if label.get('superscript'):
        text += '_' + label['superscript']
    if label.get('lexnum'):
        text += '_' + label['lexnum']
    return text


def _entry_naming_form(entry):
    addition = _text(entry.get('addtoname'))
    separator = ' ' if addition == 'se' else ''
    return addition + separator + _text(entry.get('entry_name'))


def _node_naming_form(node, fallback):
    match = re.search(r"<span class='namingform'>([^<]+)</span>", _text(node.get('lexname')))
    return html.unescape(match.group(1)) if match else fallback


def _matches(series, query, mode='contains'):
    values = series.fillna('').astype(str).str.casefold()
    query = query.casefold()
    if mode == 'exact':
        return values.eq(query)
    if mode == 'starts with':
        return values.str.startswith(query)
    return values.str.contains(re.escape(query), regex=True)


def _rows_for_index(frame, key):
    """Return zero or more rows for an index value as a DataFrame."""
    if frame is None or frame.empty or key not in frame.index:
        return pd.DataFrame(columns=[] if frame is None else frame.columns)
    rows = frame.loc[[key]]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    return rows


def _occurrence_ranges(value):
    """Return zero-based, half-open ranges from LexNet's 1-based spans."""
    ranges = []
    for start, end in re.findall(r'(\d+)\s*,\s*(\d+)', _text(value)):
        start, end = int(start) - 1, int(end) - 1
        if end > start:
            ranges.append((start, end))
    return ranges


def _example_segments(content, occurrence):
    """Convert example HTML to plain-text segments carrying occurrence marks."""
    raw = _text(content)
    ranges = _occurrence_ranges(occurrence)

    def is_marked(start, end):
        return any(start < range_end and end > range_start for range_start, range_end in ranges)

    characters = []
    index = 0
    while index < len(raw):
        if raw[index] == '<':
            end = raw.find('>', index + 1)
            if end != -1:
                characters.append((' ', False))
                index = end + 1
                continue
        if raw[index] == '&':
            match = re.match(r'&(?:#[xX][0-9a-fA-F]+|#\d+|[A-Za-z][A-Za-z0-9]+);', raw[index:])
            if match:
                end = index + len(match.group(0))
                decoded = html.unescape(match.group(0))
                marked = is_marked(index, end)
                characters.extend((character, marked) for character in decoded)
                index = end
                continue
        characters.append((raw[index], is_marked(index, index + 1)))
        index += 1

    normalized = []
    pending_space = False
    pending_mark = False
    for character, marked in characters:
        if character.isspace():
            if normalized:
                pending_space = True
                pending_mark = pending_mark or marked
            continue
        if pending_space:
            normalized.append((' ', pending_mark))
            pending_space = False
            pending_mark = False
        normalized.append((character, marked))

    segments = []
    for character, marked in normalized:
        if segments and segments[-1]['highlighted'] == marked:
            segments[-1]['text'] += character
        else:
            segments.append({'text': character, 'highlighted': marked})
    return segments


def _confidence(value):
    """Return normalized confidence, defaulting to 100 percent."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 100
    if pd.isna(number):
        return 100
    return int(number) if number.is_integer() else number


class LexnetQueries:
    """Search operations used by the GUI, kept independent of the web UI."""

    def __init__(self, data):
        self.data = data
        self.nodes = data['nodes']
        self.entries = data['entries']
        self.node_entries = self.nodes['entry_id'].to_dict()
        self.entry_labels = {
            entry_id: {
                'name': _entry_naming_form(entry) or _text(entry_id),
                'subscript': _text(entry.get('subscript')),
                'superscript': _text(entry.get('superscript')),
                'lexnum': '',
                'confidence': _confidence(entry.get('entry_%')),
            }
            for entry_id, entry in self.entries.iterrows()
        }
        self.node_labels = {}
        for node_id, node in self.nodes.iterrows():
            entry_id = node.get('entry_id')
            label = dict(self.entry_labels.get(entry_id, {
                'name': _text(node.get('std_name')) or _text(node_id),
                'subscript': '', 'superscript': '', 'lexnum': '', 'confidence': 100,
            }))
            label['name'] = _node_naming_form(node, label['name'])
            label['lexnum'] = _text(node.get('lexnum'))
            label['confidence'] = _confidence(node.get('node_%'))
            self.node_labels[node_id] = label
        self.entry_names = {key: _name_text(value) for key, value in self.entry_labels.items()}
        self.node_names = {key: _name_text(value) for key, value in self.node_labels.items()}
        self.lf_table = data['lf_names']
        self.lf_names = self.lf_table['lf_name'].fillna('').astype(str).to_dict()
        self.lf_order = {
            lexical_function_id: (
                _number(row.get('group_index')),
                _number(row.get('family_index')),
                _text(row.get('lf_name')).casefold(),
                _text(lexical_function_id),
            )
            for lexical_function_id, row in self.lf_table.iterrows()
        }
        self.feature_names = data['feature_names']['name'].fillna('').astype(str).to_dict()
        self.form_names = self._name_map(data.get('form_names'))
        self.label_names = self._name_map(data.get('label_names'))

    @staticmethod
    def _name_map(frame):
        if frame is None or frame.empty or 'name' not in frame:
            return {}
        return frame['name'].fillna('').astype(str).to_dict()

    def search_words(self, query, mode='contains', include_forms=True):
        """Find lexical entries through entry, unit, or inflected-form text."""
        query = query.strip()
        if not query:
            return pd.DataFrame(columns=EMPTY_WORD_RESULTS)

        entry_ids = set(self.entries.index[_matches(self.entries['entry_name'], query, mode)])
        node_ids = set(self.nodes.index[_matches(self.nodes['std_name'], query, mode)])
        sources = {node_id: {'unit'} for node_id in node_ids}

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
                'entry_label': self.entry_labels.get(entry_id, {}),
                'node_id': node_id,
                'std_name': self.node_names.get(node_id, _text(node.get('std_name'))),
                'unit_label': self.node_labels.get(node_id, {}),
                'pos': pos,
                'match_source': ', '.join(sorted(sources.get(node_id, {'unit'}))),
            })

        return pd.DataFrame(rows, columns=EMPTY_WORD_RESULTS).sort_values(
            ['entry_name', 'std_name'], key=lambda col: col.astype(str).str.casefold()
        ).reset_index(drop=True)

    def _resolve_names(self, queries, names):
        resolved = set()
        folded = {key: value.casefold() for key, value in names.items()}
        for query in queries:
            needle = query.casefold()
            exact = {key for key, value in folded.items() if value == needle}
            resolved.update(exact or {key for key, value in folded.items() if needle in value})
        return resolved

    def search_lexical_functions(self, names):
        """Return source/target occurrences for one or more LF names."""
        lexical_function_ids = self._resolve_names(split_query(names), self.lf_names)
        return self.search_lexical_function_ids(lexical_function_ids)

    def search_lexical_function_ids(self, lexical_function_ids):
        """Return source/target occurrences for LF IDs."""
        lexical_function_ids = set(lexical_function_ids)
        relations = self.data.get('lfs')
        if not lexical_function_ids or relations is None or relations.empty:
            return pd.DataFrame(columns=EMPTY_LF_RESULTS)

        rows = []
        matching = relations[relations['lexical_function_id'].isin(lexical_function_ids)]
        for _, relation in matching.iterrows():
            source_node_id = relation['source_node_id']
            target_node_id = relation['target_node_id']
            lexical_function_id = relation['lexical_function_id']
            rows.append({
                'lf_name': self.lf_names.get(lexical_function_id, _text(lexical_function_id)),
                'source_node_id': source_node_id,
                'source_name': self.node_names.get(source_node_id, _text(source_node_id)),
                'source_label': self.node_labels.get(source_node_id, {}),
                'target_node_id': target_node_id,
                'target_name': self.node_names.get(target_node_id, _text(target_node_id)),
                'target_label': self.node_labels.get(target_node_id, {}),
                'form': _text(relation.get('form')),
                'frame': _text(relation.get('frame')),
                'constraint': _text(relation.get('constraint')),
            })
        return pd.DataFrame(rows, columns=EMPTY_LF_RESULTS).sort_values(
            ['lf_name', 'source_name', 'target_name'],
            key=lambda col: col.astype(str).str.casefold(),
        ).reset_index(drop=True)

    def lexical_function_ids_for_family(self, family_id):
        """Return LF IDs belonging to a family, preserving XML order."""
        if 'family_id' not in self.lf_table:
            return []
        return self.lf_table.index[self.lf_table['family_id'] == family_id].tolist()

    def lexical_function_ids_for_group(self, group_index):
        """Return LF IDs belonging to a group, preserving XML order."""
        if 'group_index' not in self.lf_table:
            return []
        return self.lf_table.index[self.lf_table['group_index'] == group_index].tolist()

    def lexical_function_hierarchy(self):
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
                        {'id': _text(lexical_function_id), 'name': _text(row['lf_name'])}
                        for lexical_function_id, row in family_rows.iterrows()
                    ],
                })
            groups.append({'index': int(group_index), 'families': families})
        return groups

    def search_features(self, names, require_all=False):
        """Return units and entries carrying named grammatical features."""
        feature_ids = self._resolve_names(split_query(names), self.feature_names)
        features = self.data.get('features')
        if not feature_ids or features is None or features.empty:
            return pd.DataFrame(columns=EMPTY_FEATURE_RESULTS)

        by_node = {}
        for node_id, row in features.iterrows():
            feature_values = row.get('features', [])
            values = set(feature_values) if isinstance(feature_values, (list, tuple, set)) else set()
            pos = row.get('POS')
            if _text(pos):
                values.add(pos)
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
                'entry_label': self.entry_labels.get(entry_id, {}),
                'node_id': node_id,
                'std_name': self.node_names.get(node_id, _text(node.get('std_name'))),
                'unit_label': self.node_labels.get(node_id, {}),
                'matching_features': ', '.join(
                    sorted((self.feature_names.get(fid, _text(fid)) for fid in shown_ids), key=str.casefold)
                ),
            })
        return pd.DataFrame(rows, columns=EMPTY_FEATURE_RESULTS).sort_values(
            ['entry_name', 'std_name'], key=lambda col: col.astype(str).str.casefold()
        ).reset_index(drop=True)

    def describe_node(self, node_id):
        """Build a readable, plain-text summary of a lexical unit."""
        if node_id not in self.nodes.index:
            return f'Unknown lexical unit: {node_id}'
        node = self.nodes.loc[node_id]
        entry_id = node['entry_id']
        lines = [
            self.node_names.get(node_id, _text(node_id)),
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
        semantic_label_links = self.semantic_label_links(node_id)
        if semantic_label_links:
            lines.extend(['', 'SEMANTIC LABELS'])
            lines.extend(link['line'] for link in semantic_label_links)

        propositional_form_links = self.propositional_form_links(node_id)
        if propositional_form_links:
            lines.extend(['', 'PROPOSITIONAL FORMS'])
            lines.extend(link['line'] for link in propositional_form_links)

        relation_links = self.lexical_function_links(node_id)
        if relation_links:
            lines.extend(['', 'LEXICAL RELATIONS'])
            for direction in ('Outgoing', 'Incoming'):
                grouped_links = [link for link in relation_links if link['direction'] == direction]
                if grouped_links:
                    lines.append(direction)
                    lines.extend(link['line'] for link in grouped_links)

        example_links = self.example_links(node_id)
        if example_links:
            lines.extend(['', 'EXAMPLES'])
            lines.extend(link['line'] for link in example_links)
        return '\n'.join(lines)

    def describe_entry(self, entry_id):
        """Build a summary of an entry and its lexical units."""
        if entry_id not in self.entries.index:
            return f'Unknown lexical entry: {entry_id}'
        entry = self.entries.loc[entry_id]
        lines = [
            self.entry_names.get(entry_id, _text(entry_id)),
        ]
        metadata = []
        for column, label in [
            ('subscript', 'Subscript'), ('superscript', 'Superscript'), ('entry_status', 'Status'),
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

    def entry_unit_links(self, entry_id):
        """Return the units in an entry with compact grammatical summaries."""
        units = self.nodes[self.nodes['entry_id'] == entry_id]
        links = []
        for node_id, node in units.sort_values(
            'std_name', key=lambda values: values.fillna('').astype(str).str.casefold()
        ).iterrows():
            name = self.node_names.get(node_id, _text(node_id))
            grammar = self._node_grammar_summary(node_id)
            suffix = f'  [{grammar}]' if grammar else ''
            links.append({
                'line': ' • ' + name + suffix,
                'prefix': ' • ', 'suffix': suffix,
                'item_type': 'node', 'item_id': _text(node_id), 'item_name': name,
                'item_label': self.node_labels.get(node_id, {}),
            })
        return links

    def _node_grammar_summary(self, node_id):
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

    def inspector_payload(self, item_type, item_id):
        """Return a generic payload for the browser inspector."""
        if item_type == 'node':
            entry_id = self.node_entries.get(item_id)
            node_name = self.node_names.get(item_id, _text(item_id))
            links = [{
                'line': node_name, 'prefix': '', 'suffix': '',
                'label': self.node_labels.get(item_id, {}),
            }]
            if entry_id is not None:
                name = self.entry_names.get(entry_id, _text(entry_id))
                prefix, suffix = 'Entry: ', ''
                links.append({
                    'line': prefix + name + suffix,
                    'prefix': prefix, 'suffix': suffix,
                    'item_type': 'entry', 'item_id': _text(entry_id), 'item_name': name,
                    'item_label': self.entry_labels.get(entry_id, {}),
                })
            links.extend(self.lexical_function_links(item_id))
            links.extend(self.semantic_label_links(item_id))
            links.extend(self.propositional_form_links(item_id))
            links.extend(self.example_links(item_id))
            return {
                'title': 'Lexical unit', 'description': self.describe_node(item_id),
                'links': links,
            }
        if item_type == 'entry':
            name = self.entry_names.get(item_id, _text(item_id))
            return {
                'title': 'Lexical entry', 'description': self.describe_entry(item_id),
                'links': [{
                    'line': name, 'prefix': '', 'suffix': '',
                    'label': self.entry_labels.get(item_id, {}),
                }, *self.entry_unit_links(item_id)],
            }
        return {'title': 'Inspector', 'error': f'Unknown item type: {item_type}'}

    @staticmethod
    def _confidence_link(text, value):
        confidence = _confidence(value)
        return {
            'line': ' • ' + text,
            'prefix': ' • ',
            'suffix': '',
            'information_text': text,
            'confidence': confidence,
            'low_confidence': confidence < 100,
        }

    def semantic_label_links(self, node_id):
        """Return semantic labels with their confidence metadata."""
        links = []
        for _, row in _rows_for_index(self.data.get('labels'), node_id).iterrows():
            semantic_label_id = row.get('semantic_label_id')
            name = self.label_names.get(semantic_label_id, 'Unknown semantic label')
            if name:
                links.append(self._confidence_link(name, row.get('label_%')))
        return links

    def propositional_form_links(self, node_id):
        """Return propositional forms with their confidence metadata."""
        links = []
        for _, row in _rows_for_index(self.data.get('propforms'), node_id).iterrows():
            propform = _text(row.get('propform'))
            if propform:
                links.append(self._confidence_link(propform, row.get('propform_confid')))
        return links

    def example_links(self, node_id):
        """Return ordered examples with confidence and marked keyword spans."""
        relations = self.data.get('ex-rel')
        examples = self.data.get('examples')
        if relations is None or examples is None or relations.empty or examples.empty:
            return []

        rows = relations[relations['node_id'] == node_id]
        ordered = sorted(rows.iterrows(), key=lambda pair: _number(pair[1].get('position')))
        links = []
        for _, relation in ordered:
            example_id = relation.get('example_id')
            if example_id not in examples.index:
                continue
            example = examples.loc[example_id]
            if isinstance(example, pd.DataFrame):
                example = example.iloc[0]
            segments = _example_segments(example.get('content'), relation.get('occurrence'))
            text = ''.join(segment['text'] for segment in segments)
            if not text:
                continue
            confidence = _confidence(relation.get('%'))
            links.append({
                'line': ' • ' + text,
                'prefix': ' • ',
                'suffix': '',
                'example_segments': segments,
                'confidence': confidence,
                'low_confidence': confidence < 100,
            })
        return links

    def lexical_function_links(self, node_id):
        """Return display text and navigation targets for a unit's LF relations."""
        relations = self.data.get('lfs')
        if relations is None or relations.empty:
            return []
        links = []

        def append_groups(rows, related_id_column, direction):
            groups = {}
            for _, row in rows.iterrows():
                function_id = row['lexical_function_id']
                related_id = row[related_id_column]
                name = self.node_names.get(related_id, _text(related_id))
                item = {
                    'item_type': 'node',
                    'item_id': _text(related_id),
                    'item_name': name,
                    'item_label': self.node_labels.get(related_id, {}),
                    'frame': _text(row.get('frame')),
                    'constraint': _text(row.get('constraint')),
                    'merged': related_id_column == 'target_node_id' and _text(row.get('merged')) == '1',
                }
                groups.setdefault(function_id, []).append((_number(row.get('position')), item))
            fallback = (float('inf'), float('inf'), '', '')
            for function_id in sorted(groups, key=lambda item: self.lf_order.get(item, fallback)):
                items = [item for _, item in sorted(groups[function_id], key=lambda pair: pair[0])]
                function_name = self.lf_names.get(function_id, function_id)
                prefix = f'{function_name}: '
                def item_text(item):
                    merged = '//' if item['merged'] else ''
                    frame = f" {item['frame']}" if item['frame'] else ''
                    constraint = f" ({item['constraint']})" if item['constraint'] else ''
                    return merged + item['item_name'] + frame + constraint
                links.append({
                    'line': prefix + ', '.join(item_text(item) for item in items),
                    'prefix': prefix,
                    'suffix': '',
                    'function_name': _text(function_name),
                    'items': items,
                    'direction': direction,
                })

        append_groups(
            relations[relations['source_node_id'] == node_id],
            'target_node_id', 'Outgoing',
        )
        append_groups(
            relations[relations['target_node_id'] == node_id],
            'source_node_id', 'Incoming',
        )
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
