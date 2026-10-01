import json
import unittest
from io import BytesIO

import pandas as pd

from explorer import CSS_PATH, DAGRE_PATH, PAGE_PATH, SCRIPT_PATH, create_handler
from explorer_queries import (
    LexnetQueries, _example_segments, _propform_segments, _semantic_derivation_name,
    split_query,
)


class TestLexnetQueries(unittest.TestCase):
    def setUp(self):
        self.data = {
            'entries': pd.DataFrame([
                {'entry_id': 'e1', 'entry_name': 'chat', 'subscript': 'N, masc', 'superscript': '1', 'entry_%': 60},
                {'entry_id': 'e2', 'entry_name': 'prendre le large', 'subscript': '', 'superscript': ''},
            ]).set_index('entry_id'),
            'nodes': pd.DataFrame([
                {'node_id': 'n1', 'entry_id': 'e1', 'std_name': 'chat I', 'lexnum': 'I', 'node_%': 60},
                {'node_id': 'n2', 'entry_id': 'e1', 'std_name': 'chat II', 'lexnum': 'II'},
                {'node_id': 'n3', 'entry_id': 'e2', 'std_name': 'prendre le large', 'lexnum': ''},
            ]).set_index('node_id'),
            'forms': pd.DataFrame([
                {'node_id': 'n1', 'features': 'plural', 'signifier': 'chats'},
            ]).set_index('node_id'),
            'features': pd.DataFrame([
                {'node_id': 'n1', 'POS': 'f_noun', 'ph_str': '', 'usage': '', 'features': []},
                {'node_id': 'n3', 'POS': 'V', 'ph_str': 'idiom', 'usage': '', 'features': ['f_idiom']},
                {'node_id': 'n3', 'POS': 'V', 'ph_str': 'idiom', 'usage': '', 'features': ['f_verbal']},
            ]).set_index('node_id'),
            'feature_names': pd.DataFrame([
                {'feature_id': 'f_noun', 'name': 'noun'},
                {'feature_id': 'f_idiom', 'name': 'idiom'},
                {'feature_id': 'f_verbal', 'name': 'verbal expression'},
            ]).set_index('feature_id'),
            'label_names': pd.DataFrame([
                {
                    'semantic_label_id': 'sl1', 'name': 'Label One',
                    'status': '1', 'derivation': 'S1', 'actant_type': '0',
                    'comment': 'A label comment',
                },
            ]).set_index('semantic_label_id'),
            'label_classes': pd.DataFrame([
                {'semantic_class_id': 'c0', 'name': 'QQCH.', 'semantic_field': '0', 'inheritance_type': '0', 'comment': ''},
                {'semantic_class_id': 'c1', 'name': 'ENTITÉ', 'semantic_field': '1', 'inheritance_type': '0', 'comment': 'Things'},
            ]).set_index('semantic_class_id'),
            'label_class_edges': pd.DataFrame([
                {'parent_class_id': 'c0', 'child_class_id': 'c1'},
            ]),
            'label_memberships': pd.DataFrame([
                {'semantic_class_id': 'c1', 'semantic_label_id': 'sl1'},
            ]),
            'lf_names': pd.DataFrame([
                {'lexical_function_id': 'lf1', 'lf_name': 'Magn', 'type': 'standard', 'family_id': 'fam1', 'family_name': 'Intensity', 'group_index': 1, 'family_index': 1, 'lf_index': 1},
                {'lexical_function_id': 'lf2', 'lf_name': 'Oper1', 'type': 'standard', 'family_id': 'fam2', 'family_name': 'Support verbs', 'group_index': 2, 'family_index': 1, 'lf_index': 1},
            ]).set_index('lexical_function_id'),
            'lfs': pd.DataFrame([
                {'source_node_id': 'n1', 'lexical_function_id': 'lf1', 'target_node_id': 'n2', 'form': '', 'frame': 'N=$2', 'constraint': '', 'merged': 0},
                {'source_node_id': 'n1', 'lexical_function_id': 'lf1', 'target_node_id': 'n3', 'form': '', 'frame': '', 'constraint': 'postposé', 'merged': 1},
                {'source_node_id': 'n3', 'lexical_function_id': 'lf2', 'target_node_id': 'n1', 'form': 'prendre', 'frame': '', 'constraint': '', 'merged': 1},
            ]),
            'copolysemy_types': pd.DataFrame([
                {'cp_type': 'ct1', 'name': 'Métaphore', 'order': 2, 'semantics': 0, 'derivation': True},
                {'cp_type': 'ct2', 'name': 'Extension', 'order': 1, 'semantics': 1, 'derivation': False},
            ]).set_index('cp_type'),
            'copolysemy_subtypes': pd.DataFrame([
                {'cp_subtype': 'cs1', 'cp_type': 'ct1', 'name': 'Forme'},
            ]).set_index('cp_subtype'),
            'copolysemy': pd.DataFrame([
                {'source_node_id': 'n1', 'target_node_id': 'n2', 'cp_type': 'ct1', 'cp_subtype': 'cs1'},
                {'source_node_id': 'n2', 'target_node_id': 'n1', 'cp_type': 'ct2', 'cp_subtype': None},
            ]),
            'definitions': pd.DataFrame(columns=['node_id', 'def_HTML']).set_index('node_id'),
            'labels': pd.DataFrame([
                {'node_id': 'n1', 'semantic_label_id': 'sl1', 'label_%': 90},
            ]).set_index('node_id'),
            'propforms': pd.DataFrame([
                {
                    'node_id': 'n1', 'propform': '$1 est un $2',
                    'propform_confid': 80, 'actants': '($1=X,$2=Y)',
                },
            ]).set_index('node_id'),
            'examples': pd.DataFrame([
                {'example_id': 'x1', 'content': '<p>Le chat dort.</p>'},
                {'example_id': 'x2', 'content': 'Un chat joue.'},
            ]).set_index('example_id'),
            'ex-rel': pd.DataFrame([
                {'node_id': 'n1', 'example_id': 'x1', 'occurrence': '7,11;', 'position': 2, '%': 75},
                {'node_id': 'n1', 'example_id': 'x2', 'occurrence': '4,8;', 'position': 1, '%': 100},
            ]),
        }
        self.queries = LexnetQueries(self.data)

    def test_split_query_accepts_commas_and_newlines(self):
        self.assertEqual(split_query('Magn, Oper1\nMagn'), ['Magn', 'Oper1'])

    def test_word_search_returns_units_from_matching_entry(self):
        result = self.queries.search_words('chat', mode='exact')
        self.assertEqual(set(result.node_id), {'n1', 'n2'})
        self.assertEqual(set(result.match_source), {'entry'})

    def test_word_search_can_match_an_inflected_form(self):
        result = self.queries.search_words('chats', mode='exact')
        self.assertEqual(result.node_id.tolist(), ['n1'])
        self.assertEqual(result.iloc[0].match_source, 'form')

    def test_lexical_function_search_accepts_a_list(self):
        result = self.queries.search_lexical_functions('Magn, Oper1')
        self.assertEqual(set(result.lf_name), {'Magn', 'Oper1'})
        self.assertEqual(set(result.source_name), {'chat_N, masc_1_I', 'prendre le large'})
        self.assertEqual(result.iloc[0].source_label['subscript'], 'N, masc')

    def test_lexical_function_hierarchy_clusters_families_by_group(self):
        hierarchy = self.queries.lexical_function_hierarchy()
        self.assertEqual([group['index'] for group in hierarchy], [1, 2])
        self.assertEqual(hierarchy[0]['families'][0]['name'], 'Intensity')
        self.assertEqual(hierarchy[0]['families'][0]['functions'][0]['name'], 'Magn')
        self.assertEqual(self.queries.lexical_function_ids_for_family('fam2'), ['lf2'])
        self.assertEqual(self.queries.lexical_function_ids_for_group(1), ['lf1'])

    def test_feature_search_supports_any_and_all(self):
        any_result = self.queries.search_features('idiom, noun')
        all_result = self.queries.search_features('idiom, verbal expression', require_all=True)
        self.assertEqual(set(any_result.node_id), {'n1', 'n3'})
        self.assertEqual(all_result.node_id.tolist(), ['n3'])

    def test_semantic_label_search_supports_labels_and_class_descendants(self):
        by_label = self.queries.search_semantic_labels(query='Label One')
        by_root = self.queries.search_semantic_labels(class_id='c0', include_descendants=True)
        direct_root = self.queries.search_semantic_labels(class_id='c0', include_descendants=False)

        self.assertEqual(by_label.node_id.tolist(), ['n1'])
        self.assertEqual(by_root.semantic_label_id.tolist(), ['sl1'])
        self.assertTrue(direct_root.empty)
        self.assertEqual(by_label.iloc[0].confidence, 90)

    def test_semantic_hierarchy_and_inspectors_are_navigable(self):
        hierarchy = self.queries.semantic_class_hierarchy()
        self.assertEqual([(item['name'], item['depth']) for item in hierarchy], [
            ('QQCH.', 0), ('ENTITÉ', 1),
        ])
        self.assertTrue(all('labels' not in item for item in hierarchy))
        label_payload = self.queries.inspector_payload('semantic_label', 'sl1')
        class_payload = self.queries.inspector_payload('semantic_class', 'c0')
        self.assertIn('Derivation: S1', label_payload['description'])
        self.assertEqual(label_payload['links'][0]['item_id'], 'c1')
        self.assertEqual(label_payload['semantic_units']['count'], 1)
        self.assertIn('Inheritance: simple', class_payload['description'])
        self.assertIn('DESCENDANT CLASSES', class_payload['description'])
        self.assertIn(' • S_1: Label One', class_payload['description'])
        self.assertNotIn('LEXICAL UNITS', class_payload['description'])
        self.assertNotIn('semantic_units', class_payload)
        label_link = next(link for link in class_payload['links'] if link['item_type'] == 'semantic_label')
        self.assertEqual(label_link['function_name'], 'S_1')
        self.assertEqual(
            {link['item_type'] for link in class_payload['links']},
            {'semantic_class', 'semantic_label'},
        )
        self.assertEqual(
            self.queries.semantic_unit_results('semantic_class', 'c0')['node_id'].tolist(),
            ['n1'],
        )

    def test_semantic_derivations_are_normalized_as_lf_names(self):
        self.assertEqual(_semantic_derivation_name('S1'), 'S_1')
        self.assertEqual(_semantic_derivation_name('A2Manif'), 'A_2Manif')
        self.assertEqual(_semantic_derivation_name('Convij'), 'Conv_ij')
        self.assertEqual(_semantic_derivation_name('V0Convij'), 'V_0Conv_ij')
        self.assertEqual(_semantic_derivation_name('---'), '')

    def test_node_description_resolves_feature_names(self):
        description = self.queries.describe_node('n3')
        self.assertIn('prendre le large', description)
        self.assertIn('Features: idiom', description)
        self.assertIn('Oper1', description)

    def test_node_description_uses_label_text_and_hides_internal_ids(self):
        description = self.queries.describe_node('n1')
        self.assertIn('Label One', description)
        self.assertNotIn('sl1', description)
        self.assertNotIn('Node: n1', description)
        self.assertNotIn('(e1)', description)

    def test_lexical_function_links_group_alternative_values(self):
        links = self.queries.lexical_function_links('n1')
        magn = links[0]
        self.assertEqual(magn['line'], 'Magn: chat_N, masc_1_II N=$2, //prendre le large (postposé)')
        self.assertEqual(magn['function_name'], 'Magn')
        self.assertEqual(magn['direction'], 'Outgoing')
        self.assertEqual([item['item_id'] for item in magn['items']], ['n2', 'n3'])
        self.assertEqual(magn['items'][0]['frame'], 'N=$2')
        self.assertEqual(magn['items'][1]['constraint'], 'postposé')
        self.assertFalse(magn['items'][0]['merged'])
        self.assertTrue(magn['items'][1]['merged'])
        self.assertEqual(magn['line'].count('Magn'), 1)
        self.assertEqual(links[1]['direction'], 'Incoming')
        self.assertEqual(links[1]['items'][0]['item_id'], 'n3')
        self.assertFalse(links[1]['items'][0]['merged'])

    def test_lexical_function_links_follow_model_and_value_order(self):
        data = dict(self.data)
        data['lf_names'] = pd.DataFrame([
            {'lexical_function_id': 'late', 'lf_name': 'Late', 'group_index': 2, 'family_index': 1},
            {'lexical_function_id': 'zeta', 'lf_name': 'Zeta', 'group_index': 1, 'family_index': 1},
            {'lexical_function_id': 'alpha', 'lf_name': 'Alpha', 'group_index': 1, 'family_index': 1},
            {'lexical_function_id': 'family2', 'lf_name': 'FamilyTwo', 'group_index': 1, 'family_index': 2},
        ]).set_index('lexical_function_id')
        data['lfs'] = pd.DataFrame([
            {'source_node_id': 'n1', 'lexical_function_id': 'late', 'target_node_id': 'n2', 'position': 1},
            {'source_node_id': 'n1', 'lexical_function_id': 'zeta', 'target_node_id': 'n2', 'position': 2},
            {'source_node_id': 'n1', 'lexical_function_id': 'family2', 'target_node_id': 'n2', 'position': 1},
            {'source_node_id': 'n1', 'lexical_function_id': 'alpha', 'target_node_id': 'n2', 'position': 1},
            {'source_node_id': 'n1', 'lexical_function_id': 'zeta', 'target_node_id': 'n3', 'position': 1},
        ])

        links = LexnetQueries(data).lexical_function_links('n1')

        self.assertEqual([link['function_name'] for link in links], ['Alpha', 'Zeta', 'FamilyTwo', 'Late'])
        self.assertEqual([item['item_id'] for item in links[1]['items']], ['n3', 'n2'])

    def test_lexical_function_names_use_source_actant_variables(self):
        data = dict(self.data)
        data['lf_names'] = pd.concat([
            self.data['lf_names'],
            pd.DataFrame([{
                'lexical_function_id': 'lf3', 'lf_name': '$1=‘chanteurs’',
                'group_index': 3, 'family_index': 1,
            }]).set_index('lexical_function_id'),
        ])
        data['lfs'] = pd.DataFrame([{
            'source_node_id': 'n1', 'lexical_function_id': 'lf3',
            'target_node_id': 'n2', 'position': 1,
        }])
        queries = LexnetQueries(data)

        outgoing = queries.lexical_function_links('n1')[0]
        incoming = queries.lexical_function_links('n2')[0]
        result = queries.search_lexical_function_ids(['lf3']).iloc[0]

        self.assertEqual(outgoing['function_name'], 'X=‘chanteurs’')
        self.assertEqual(incoming['function_name'], 'X=‘chanteurs’')
        self.assertEqual(result.lf_name, 'X=‘chanteurs’')
        self.assertEqual(outgoing['function_segments'][0], {'text': 'X', 'actant': '$1'})
        self.assertEqual(queries.lf_display_names['lf3'], 'X=‘chanteurs’')
        self.assertEqual(
            queries.search_lexical_functions('X=‘chanteurs’').iloc[0].lf_name,
            'X=‘chanteurs’',
        )

    def test_entry_inspector_lists_units_with_semantic_information(self):
        payload = self.queries.inspector_payload('entry', 'e1')
        self.assertEqual(payload['title'], 'Lexical entry')
        self.assertEqual({link.get('item_id') for link in payload['links']} - {None}, {'n1', 'n2'})
        self.assertIn('chat_N, masc_1_I (Label One : X est un Y)', payload['description'])
        self.assertEqual(payload['links'][0]['label']['name'], 'chat')
        self.assertEqual(payload['links'][1]['item_label']['lexnum'], 'I')
        self.assertEqual(payload['links'][1]['unit_annotations']['labels'][0]['text'], 'Label One')
        self.assertEqual(
            payload['links'][1]['unit_annotations']['propforms'][0]['text'],
            'X est un Y',
        )
        self.assertGreater(
            payload['description'].index('COPOLYSEMY GRAPH'),
            payload['description'].index('LEXICAL UNITS'),
        )

    def test_entry_inspector_exposes_directed_copolysemy_graph(self):
        graph = self.queries.inspector_payload('entry', 'e1')['copolysemy_graph']

        self.assertEqual([node['item_id'] for node in graph['nodes']], ['n1', 'n2'])
        self.assertEqual(len(graph['edges']), 2)
        self.assertEqual(graph['edges'][0]['type_name'], 'Métaphore')
        self.assertEqual(graph['edges'][0]['subtype_name'], 'Forme')
        self.assertFalse(graph['edges'][1]['derivation'])
        self.assertEqual(
            {(edge['source_id'], edge['target_id']) for edge in graph['edges']},
            {('n1', 'n2'), ('n2', 'n1')},
        )

    def test_node_inspector_links_back_to_its_entry(self):
        payload = self.queries.inspector_payload('node', 'n1')
        entry_link = next(link for link in payload['links'] if link.get('item_type') == 'entry')
        self.assertEqual(entry_link['item_type'], 'entry')
        self.assertEqual(entry_link['item_id'], 'e1')
        self.assertEqual(entry_link['item_label']['confidence'], 60)
        self.assertEqual(payload['links'][0]['label']['confidence'], 60)

    def test_information_confidence_is_exposed_consistently(self):
        payload = self.queries.inspector_payload('node', 'n1')
        label = next(link for link in payload['links'] if link.get('information_text') == 'Label One')
        propform = next(link for link in payload['links'] if link.get('information_text') == 'X est un Y')

        self.assertTrue(label['low_confidence'])
        self.assertEqual(label['confidence'], 90)
        self.assertTrue(propform['low_confidence'])
        self.assertEqual(propform['confidence'], 80)
        self.assertEqual(
            [segment.get('actant') for segment in propform['information_segments'] if segment.get('actant')],
            ['$1', '$2'],
        )

    def test_propform_segments_support_subactants_and_generic_references(self):
        segments = _propform_segments(
            '$1 agit sur $2, surtout $2.2',
            '($1=X,$2.1=Y1,$2.2=Y2)',
        )

        self.assertEqual(''.join(segment['text'] for segment in segments), 'X agit sur Y, surtout Y2')
        self.assertEqual(
            [segment['actant'] for segment in segments if 'actant' in segment],
            ['$1', '$2', '$2.2'],
        )
        mismatched = _propform_segments('~ sur $1', '($2=X)')
        self.assertEqual(''.join(segment['text'] for segment in mismatched), '~ sur X')

    def test_examples_follow_position_and_mark_occurrences(self):
        links = self.queries.example_links('n1')

        self.assertEqual([link['line'] for link in links], [
            ' • Un chat joue.', ' • Le chat dort.',
        ])
        self.assertEqual(
            [segment['text'] for segment in links[0]['example_segments'] if segment['highlighted']],
            ['chat'],
        )
        self.assertFalse(links[0]['low_confidence'])
        self.assertTrue(links[1]['low_confidence'])
        self.assertEqual(links[1]['confidence'], 75)

    def test_example_segments_support_multiple_spans_and_html_entities(self):
        segments = _example_segments('<p>chat&nbsp;et chat</p>', '4,8;17,21;')

        self.assertEqual(''.join(segment['text'] for segment in segments), 'chat et chat')
        self.assertEqual(
            [segment['text'] for segment in segments if segment['highlighted']],
            ['chat', 'chat'],
        )

    def test_browser_asset_is_available(self):
        self.assertTrue(PAGE_PATH.is_file())
        self.assertTrue(CSS_PATH.is_file())
        self.assertTrue(SCRIPT_PATH.is_file())
        self.assertTrue(DAGRE_PATH.is_file())
        page = PAGE_PATH.read_text(encoding='utf8')
        script = SCRIPT_PATH.read_text(encoding='utf8')
        self.assertIn('<title>LexNet Explorer</title>', page)
        self.assertIn('explorer.css', page)
        self.assertIn('explorer.js', page)
        self.assertIn('vendor/dagre.min.js', page)
        self.assertIn('Semantic hierarchy', page)
        self.assertIn('semantic-browser', page)
        self.assertIn('main-resizer', page)
        self.assertIn('inspector-back', page)
        self.assertIn('inspector-history-toggle', page)
        self.assertIn('inspector-history-menu', page)
        self.assertIn('inspector-section', script)
        self.assertIn("document.createElement('ul')", script)
        self.assertIn('lf-values', script)
        self.assertIn('lf-name', script)
        self.assertIn('lf-name-scripts', script)
        self.assertIn('lf-frame', script)
        self.assertIn('lf-constraint', script)
        self.assertIn('lf-merge', script)
        self.assertIn('function appendLexicalName', script)
        self.assertIn("title !== 'ENTRY INFORMATION'", script)
        self.assertIn('lexical-name-sense', script)
        self.assertIn('lexical-name-scripts', script)
        self.assertIn('COPOLYSEMY GRAPH', script)
        self.assertIn('copolysemy-graph', script)
        self.assertIn('inspectorHistoryLimit = 100', script)
        self.assertIn('inspectorHistoryMenu.replaceChildren()', script)
        self.assertNotIn('semantic-unit-label', script)
        self.assertIn('example-occurrence', script)
        self.assertIn("atomicLfNames = ['De_nouveau']", script)
        self.assertIn("document.createElement(scriptMarker === '_' ? 'sub' : 'sup')", script)

    def test_http_handler_serves_page_and_search_api(self):
        handler_class = create_handler(self.queries, '/tmp/example')

        def get(path):
            handler = handler_class.__new__(handler_class)
            handler.path = path
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.requestline = f'GET {path} HTTP/1.1'
            handler.wfile = BytesIO()
            handler.do_GET()
            headers, body = handler.wfile.getvalue().split(b'\r\n\r\n', 1)
            header_lines = headers.decode('iso-8859-1').splitlines()
            status = int(header_lines[0].split()[1])
            content_type = next(
                line.split(':', 1)[1].strip().split(';', 1)[0]
                for line in header_lines if line.lower().startswith('content-type:')
            )
            self.assertEqual(status, 200)
            return status, content_type, body

        _, _, page_body = get('/')
        page = page_body.decode('utf-8')
        _, stylesheet_type, stylesheet_body = get('/explorer.css')
        stylesheet = stylesheet_body.decode('utf-8')
        _, script_type, script_body = get('/explorer.js')
        script = script_body.decode('utf-8')
        _, dagre_type, dagre_body = get('/vendor/dagre.min.js')
        _, _, meta_body = get('/api/meta')
        meta_payload = json.loads(meta_body)
        _, _, search_body = get('/api/search?kind=word&q=chat&mode=exact&forms=1')
        payload = json.loads(search_body)
        _, _, family_body = get('/api/search?kind=lf&family_id=fam1')
        family_payload = json.loads(family_body)
        _, _, group_body = get('/api/search?kind=lf&family_id=group%3A2')
        group_payload = json.loads(group_body)
        _, _, semantic_body = get('/api/search?kind=semantic&class_id=c0&descendants=1')
        semantic_payload = json.loads(semantic_body)
        _, _, node_body = get('/api/inspect?type=node&id=n3')
        node_payload = json.loads(node_body)
        _, _, entry_body = get('/api/inspect?type=entry&id=e1')
        entry_payload = json.loads(entry_body)
        _, _, semantic_units_body = get('/api/semantic-units?type=semantic_label&id=sl1&limit=1')
        semantic_units_payload = json.loads(semantic_units_body)

        self.assertIn('LexNet Explorer', page)
        self.assertIn(':root', stylesheet)
        self.assertEqual(stylesheet_type, 'text/css')
        self.assertIn('function sortBy(key)', script)
        self.assertEqual(script_type, 'text/javascript')
        self.assertEqual(dagre_type, 'text/javascript')
        self.assertIn(b'graphlib', dagre_body)
        self.assertNotIn('semantic_labels', meta_payload)
        self.assertEqual(meta_payload['semantic_hierarchy'][0]['count'], 1)
        self.assertTrue(meta_payload['semantic_hierarchy'][1]['semantic_field'])
        self.assertEqual({row['node_id'] for row in payload['rows']}, {'n1', 'n2'})
        self.assertEqual(payload['rows'][0]['entry_label']['subscript'], 'N, masc')
        self.assertEqual([row['lf_name'] for row in family_payload['rows']], ['Magn', 'Magn'])
        self.assertIsInstance(family_payload['rows'][0]['lf_name_segments'], list)
        self.assertEqual([row['lf_name'] for row in group_payload['rows']], ['Oper1'])
        self.assertEqual(semantic_payload['rows'][0]['semantic_label_id'], 'sl1')
        incoming = next(link for link in node_payload['links'] if link.get('direction') == 'Incoming')
        self.assertEqual(incoming['items'][0]['item_id'], 'n1')
        self.assertEqual(entry_payload['title'], 'Lexical entry')
        self.assertEqual(semantic_units_payload['count'], 1)
        self.assertEqual(semantic_units_payload['rows'][0]['node_id'], 'n1')


if __name__ == '__main__':
    unittest.main()
