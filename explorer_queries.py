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
    'source_label', 'target_label', 'form', 'frame', 'constraint', 'lf_name_segments'
]
EMPTY_FEATURE_RESULTS = [
    'entry_id', 'entry_name', 'entry_label', 'node_id', 'std_name', 'unit_label',
    'matching_features'
]
EMPTY_LABEL_RESULTS = [
    'semantic_label_id', 'semantic_label_name', 'class_names', 'entry_id',
    'entry_name', 'entry_label', 'node_id', 'std_name', 'unit_label', 'confidence',
]
DEFAULT_ACTANT_LABELS = {'1': 'X', '2': 'Y', '3': 'Z', '4': 'W', '5': 'V', '6': 'U'}


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


def _actant_labels(actants):
    """Parse a LexNet actant list into code-to-variable mappings."""
    labels = dict(re.findall(
        r'\$(\d+(?:\.\d+)?)\s*=\s*([^,)]*)',
        _text(actants),
    ))

    # Some forms use a generic $i while the list only defines $i.1, $i.2, etc.
    # In that case Y1/Y2 unambiguously imply the generic semantic variable Y.
    for key in list(labels):
        if '.' not in key:
            continue
        base = key.split('.', 1)[0]
        if base in labels:
            continue
        sublabels = [label for item, label in labels.items() if item.startswith(base + '.')]
        stems = {re.sub(r'\d+$', '', label) for label in sublabels}
        if len(stems) == 1:
            labels[base] = stems.pop()
    return labels


def _actant_segments(text, labels):
    """Replace actant codes in text and retain their identity for markup."""
    text = _text(text)
    segments = []
    position = 0
    for match in re.finditer(r'\$(\d+(?:\.\d+)?)', text):
        if match.start() > position:
            segments.append({'text': text[position:match.start()]})
        key = match.group(1)
        label = labels.get(key)
        if label:
            segments.append({'text': label, 'actant': '$' + key})
        else:
            segments.append({'text': match.group(0)})
        position = match.end()
    if position < len(text):
        segments.append({'text': text[position:]})
    return segments


def _propform_segments(propform, actants):
    """Replace propform actant codes with their textual semantic variables."""
    labels = _actant_labels(actants)

    placeholders = set(re.findall(r'\$(\d+(?:\.\d+)?)', _text(propform)))
    if len(placeholders) == 1 and len(set(labels.values())) == 1:
        # Two source rows disagree on $1 versus $2, but each has only one
        # possible semantic variable, so the intended display is unambiguous.
        labels.setdefault(placeholders.pop(), next(iter(labels.values())))
    return _actant_segments(propform, labels)


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
        self.lf_display_names = {
            function_id: ''.join(
                segment['text'] for segment in _actant_segments(name, DEFAULT_ACTANT_LABELS)
            )
            for function_id, name in self.lf_names.items()
        }
        self.node_actants = {
            node_id: _actant_labels(row.get('actants'))
            for node_id, row in data.get('propforms', pd.DataFrame()).iterrows()
        }
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
        self.label_table = data.get('label_names', pd.DataFrame())
        self.label_names = self._name_map(self.label_table)
        self.label_classes = data.get('label_classes', pd.DataFrame())
        self.label_class_edges = data.get('label_class_edges', pd.DataFrame())
        self.label_memberships = data.get('label_memberships', pd.DataFrame())
        self.class_children = {}
        self.class_parents = {}
        if not self.label_class_edges.empty:
            for _, edge in self.label_class_edges.iterrows():
                parent_id, child_id = edge['parent_class_id'], edge['child_class_id']
                self.class_children.setdefault(parent_id, []).append(child_id)
                self.class_parents.setdefault(child_id, []).append(parent_id)
        self.class_labels = {}
        self.label_class_ids = {}
        if not self.label_memberships.empty:
            for _, membership in self.label_memberships.iterrows():
                class_id = membership['semantic_class_id']
                label_id = membership['semantic_label_id']
                self.class_labels.setdefault(class_id, []).append(label_id)
                self.label_class_ids.setdefault(label_id, []).append(class_id)
        label_relations = data.get('labels')
        self.label_assignment_counts = (
            label_relations['semantic_label_id'].value_counts().to_dict()
            if label_relations is not None and not label_relations.empty else {}
        )

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
        queries = split_query(names)
        lexical_function_ids = self._resolve_names(queries, self.lf_names)
        lexical_function_ids.update(self._resolve_names(queries, self.lf_display_names))
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
            function_segments = self._lexical_function_name_segments(
                lexical_function_id, source_node_id,
            )
            rows.append({
                'lf_name': ''.join(segment['text'] for segment in function_segments),
                'lf_name_segments': function_segments,
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

    def _lexical_function_name_segments(self, function_id, source_node_id):
        """Return an LF name with actants resolved for its source unit."""
        labels = {**DEFAULT_ACTANT_LABELS, **self.node_actants.get(source_node_id, {})}
        name = self.lf_names.get(function_id, _text(function_id))
        return _actant_segments(name, labels)

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
                        {
                            'id': _text(lexical_function_id),
                            'name': self.lf_display_names.get(
                                lexical_function_id, _text(row['lf_name']),
                            ),
                        }
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

    def semantic_class_hierarchy(self):
        """Return one row per class path, preserving multiple inheritance."""
        if self.label_classes.empty:
            return []
        child_ids = set(self.class_parents)
        roots = [class_id for class_id in self.label_classes.index if class_id not in child_ids]
        rows = []

        def visit(class_id, depth, ancestors):
            if class_id in ancestors or class_id not in self.label_classes.index:
                return
            row = self.label_classes.loc[class_id]
            direct_labels = sorted(
                ({
                    'id': _text(label_id),
                    'name': self.label_names.get(label_id, _text(label_id)),
                    'count': int(self.label_assignment_counts.get(label_id, 0)),
                }
                 for label_id in self.class_labels.get(class_id, [])),
                key=lambda item: item['name'].casefold(),
            )
            descendant_labels = self.semantic_label_ids_for_class(class_id, True)
            rows.append({
                'id': _text(class_id),
                'name': _text(row.get('name')),
                'depth': depth,
                'semantic_field': _text(row.get('semantic_field')) == '1',
                'inheritance_type': _text(row.get('inheritance_type')),
                'labels': direct_labels,
                'count': int(sum(self.label_assignment_counts.get(item, 0) for item in descendant_labels)),
                'has_children': bool(self.class_children.get(class_id) or direct_labels),
                'multiple_paths': len(self.class_parents.get(class_id, [])) > 1,
            })
            next_ancestors = ancestors | {class_id}
            children = sorted(
                set(self.class_children.get(class_id, [])),
                key=lambda item: _text(self.label_classes.loc[item].get('name')).casefold(),
            )
            for child_id in children:
                visit(child_id, depth + 1, next_ancestors)

        for root_id in roots:
            visit(root_id, 0, set())
        return rows

    def semantic_label_ids_for_class(self, class_id, include_descendants=True):
        """Return labels directly or recursively contained by a semantic class."""
        class_ids = {class_id}
        if include_descendants:
            pending = [class_id]
            while pending:
                current = pending.pop()
                for child_id in self.class_children.get(current, []):
                    if child_id not in class_ids:
                        class_ids.add(child_id)
                        pending.append(child_id)
        return {
            label_id for item in class_ids
            for label_id in self.class_labels.get(item, [])
        }

    def search_semantic_labels(
        self, query='', class_id='', semantic_label_id='', include_descendants=True,
    ):
        """Find lexical units through labels or semantic classes."""
        label_ids = set()
        query = query.strip()
        if semantic_label_id:
            label_ids.add(semantic_label_id)
        elif class_id:
            label_ids = self.semantic_label_ids_for_class(class_id, include_descendants)
        elif query:
            label_ids.update(self._resolve_names([query], self.label_names))
            class_names = self._name_map(self.label_classes)
            for matching_class_id in self._resolve_names([query], class_names):
                label_ids.update(self.semantic_label_ids_for_class(
                    matching_class_id, include_descendants,
                ))
        if not label_ids:
            return pd.DataFrame(columns=EMPTY_LABEL_RESULTS)

        labels = self.data.get('labels')
        if labels is None or labels.empty:
            return pd.DataFrame(columns=EMPTY_LABEL_RESULTS)
        rows = []
        matching = labels[labels['semantic_label_id'].isin(label_ids)]
        for node_id, relation in matching.iterrows():
            if node_id not in self.nodes.index:
                continue
            label_id = relation['semantic_label_id']
            node = self.nodes.loc[node_id]
            entry_id = node['entry_id']
            class_names = sorted({
                _text(self.label_classes.loc[item].get('name'))
                for item in self.label_class_ids.get(label_id, [])
                if item in self.label_classes.index
            }, key=str.casefold)
            rows.append({
                'semantic_label_id': label_id,
                'semantic_label_name': self.label_names.get(label_id, _text(label_id)),
                'class_names': ', '.join(class_names),
                'entry_id': entry_id,
                'entry_name': self.entry_names.get(entry_id, _text(entry_id)),
                'entry_label': self.entry_labels.get(entry_id, {}),
                'node_id': node_id,
                'std_name': self.node_names.get(node_id, _text(node_id)),
                'unit_label': self.node_labels.get(node_id, {}),
                'confidence': _confidence(relation.get('label_%')),
            })
        return pd.DataFrame(rows, columns=EMPTY_LABEL_RESULTS).sort_values(
            ['semantic_label_name', 'entry_name', 'std_name'],
            key=lambda column: column.astype(str).str.casefold(),
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
            label_links = self.semantic_label_links(node_id)
            propform_links = self.propositional_form_links(node_id)
            annotations = {
                'labels': [{
                    'text': link['information_text'],
                    'item_id': link['item_id'],
                    'confidence': link['confidence'],
                    'low_confidence': link['low_confidence'],
                } for link in label_links],
                'propforms': [{
                    'text': link['information_text'],
                    'segments': link['information_segments'],
                    'confidence': link['confidence'],
                    'low_confidence': link['low_confidence'],
                } for link in propform_links],
            }
            annotation_groups = [
                ', '.join(item['text'] for item in annotations['labels']),
                ', '.join(item['text'] for item in annotations['propforms']),
            ]
            annotation_text = ' : '.join(filter(None, annotation_groups))
            suffix = f' ({annotation_text})' if annotation_text else ''
            links.append({
                'line': ' • ' + name + suffix,
                'prefix': ' • ', 'suffix': suffix,
                'item_type': 'node', 'item_id': _text(node_id), 'item_name': name,
                'item_label': self.node_labels.get(node_id, {}),
                'unit_annotations': annotations if annotation_text else None,
            })
        return links

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
        if item_type == 'semantic_label':
            return self.semantic_label_payload(item_id)
        if item_type == 'semantic_class':
            return self.semantic_class_payload(item_id)
        return {'title': 'Inspector', 'error': f'Unknown item type: {item_type}'}

    @staticmethod
    def _plain_link(text, item_type, item_id, prefix=' • '):
        return {
            'line': prefix + text,
            'prefix': prefix,
            'suffix': '',
            'plain_item_label': text,
            'item_type': item_type,
            'item_id': _text(item_id),
        }

    def semantic_label_payload(self, semantic_label_id):
        """Return an inspector payload for one semantic-label instance."""
        if semantic_label_id not in self.label_names:
            return {'title': 'Semantic label', 'error': 'Unknown semantic label'}
        name = self.label_names[semantic_label_id]
        lines = [name]
        links = []
        metadata = self.label_table.loc[semantic_label_id] if semantic_label_id in self.label_table.index else {}
        information = []
        for column, label in [
            ('derivation', 'Derivation'), ('actant_type', 'Actant type'),
            ('status', 'Status'), ('comment', 'Comment'),
        ]:
            value = _text(metadata.get(column))
            if value:
                information.append(f' • {label}: {value}')
        if information:
            lines.extend(['', 'LABEL INFORMATION', *information])

        class_links = []
        for class_id in self.label_class_ids.get(semantic_label_id, []):
            if class_id not in self.label_classes.index:
                continue
            class_name = _text(self.label_classes.loc[class_id].get('name'))
            class_links.append(self._plain_link(class_name, 'semantic_class', class_id))
        if class_links:
            lines.extend(['', 'CLASSIFICATION', *(link['line'] for link in class_links)])
            links.extend(class_links)
        return {'title': 'Semantic label', 'description': '\n'.join(lines), 'links': links}

    def semantic_class_payload(self, class_id):
        """Return an inspector payload for one semantic class."""
        if self.label_classes.empty or class_id not in self.label_classes.index:
            return {'title': 'Semantic class', 'error': 'Unknown semantic class'}
        row = self.label_classes.loc[class_id]
        name = _text(row.get('name'))
        lines = [name]
        links = []
        inheritance = {
            '0': 'simple', '1': 'multiple inclusive', '2': 'multiple exclusive',
        }.get(_text(row.get('inheritance_type')), _text(row.get('inheritance_type')))
        information = [f' • Inheritance: {inheritance}']
        if _text(row.get('semantic_field')) == '1':
            information.append(' • Semantic field')
        comment = _text(row.get('comment'))
        if comment:
            information.append(' • Comment: ' + re.sub(r'\s+', ' ', comment).strip())
        lines.extend(['', 'CLASS INFORMATION', *information])

        sections = [
            ('PARENTS', self.class_parents.get(class_id, []), 'semantic_class'),
            ('SUBCLASSES', self.class_children.get(class_id, []), 'semantic_class'),
            ('DIRECT LABELS', self.class_labels.get(class_id, []), 'semantic_label'),
        ]
        for title, item_ids, item_type in sections:
            section_links = []
            for item_id in sorted(set(item_ids), key=lambda item: (
                self.label_names.get(item, '') if item_type == 'semantic_label'
                else _text(self.label_classes.loc[item].get('name'))
            ).casefold()):
                item_name = (
                    self.label_names.get(item_id, _text(item_id))
                    if item_type == 'semantic_label'
                    else _text(self.label_classes.loc[item_id].get('name'))
                )
                section_links.append(self._plain_link(item_name, item_type, item_id))
            if section_links:
                lines.extend(['', title, *(link['line'] for link in section_links)])
                links.extend(section_links)
        return {'title': 'Semantic class', 'description': '\n'.join(lines), 'links': links}

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
                link = self._confidence_link(name, row.get('label_%'))
                link.update({
                    'plain_item_label': name,
                    'item_type': 'semantic_label',
                    'item_id': _text(semantic_label_id),
                })
                links.append(link)
        return links

    def propositional_form_links(self, node_id):
        """Return propositional forms with their confidence metadata."""
        links = []
        for _, row in _rows_for_index(self.data.get('propforms'), node_id).iterrows():
            propform = _text(row.get('propform'))
            if propform:
                segments = _propform_segments(propform, row.get('actants'))
                text = ''.join(segment['text'] for segment in segments)
                link = self._confidence_link(text, row.get('propform_confid'))
                link['information_segments'] = segments
                links.append(link)
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
                function_segments = self._lexical_function_name_segments(
                    function_id, row['source_node_id'],
                )
                function_name = ''.join(segment['text'] for segment in function_segments)
                group_key = (function_id, function_name)
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
                group = groups.setdefault(group_key, {
                    'segments': function_segments,
                    'items': [],
                })
                group['items'].append((_number(row.get('position')), item))
            fallback = (float('inf'), float('inf'), '', '')
            for function_id, function_name in sorted(
                groups, key=lambda item: self.lf_order.get(item[0], fallback),
            ):
                group = groups[(function_id, function_name)]
                items = [item for _, item in sorted(group['items'], key=lambda pair: pair[0])]
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
                    'function_segments': group['segments'],
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
