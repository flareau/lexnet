import json
import threading
import unittest
from urllib.request import urlopen

import pandas as pd

from explorer import CSS_PATH, PAGE_PATH, SCRIPT_PATH, create_server
from explorer_queries import LexnetQueries, _example_segments, split_query


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
                {'semantic_label_id': 'sl1', 'name': 'Label One'},
            ]).set_index('semantic_label_id'),
            'lf_names': pd.DataFrame([
                {'lexical_function_id': 'lf1', 'lf_name': 'Magn', 'type': 'standard', 'family_id': 'fam1', 'family_name': 'Intensity', 'group_index': 1, 'family_index': 1, 'lf_index': 1},
                {'lexical_function_id': 'lf2', 'lf_name': 'Oper1', 'type': 'standard', 'family_id': 'fam2', 'family_name': 'Support verbs', 'group_index': 2, 'family_index': 1, 'lf_index': 1},
            ]).set_index('lexical_function_id'),
            'lfs': pd.DataFrame([
                {'source_node_id': 'n1', 'lexical_function_id': 'lf1', 'target_node_id': 'n2', 'form': '', 'frame': 'N=$2', 'constraint': '', 'merged': 0},
                {'source_node_id': 'n1', 'lexical_function_id': 'lf1', 'target_node_id': 'n3', 'form': '', 'frame': '', 'constraint': 'postposé', 'merged': 1},
                {'source_node_id': 'n3', 'lexical_function_id': 'lf2', 'target_node_id': 'n1', 'form': 'prendre', 'frame': '', 'constraint': '', 'merged': 1},
            ]),
            'definitions': pd.DataFrame(columns=['node_id', 'def_HTML']).set_index('node_id'),
            'labels': pd.DataFrame([
                {'node_id': 'n1', 'semantic_label_id': 'sl1', 'label_%': 90},
            ]).set_index('node_id'),
            'propforms': pd.DataFrame([
                {'node_id': 'n1', 'propform': 'X est un chat', 'propform_confid': 80},
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

    def test_entry_inspector_lists_units_with_grammar(self):
        payload = self.queries.inspector_payload('entry', 'e1')
        self.assertEqual(payload['title'], 'Lexical entry')
        self.assertEqual({link.get('item_id') for link in payload['links']} - {None}, {'n1', 'n2'})
        self.assertIn('chat_N, masc_1_I  [noun]', payload['description'])
        self.assertEqual(payload['links'][0]['label']['name'], 'chat')
        self.assertEqual(payload['links'][1]['item_label']['lexnum'], 'I')

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
        propform = next(link for link in payload['links'] if link.get('information_text') == 'X est un chat')

        self.assertTrue(label['low_confidence'])
        self.assertEqual(label['confidence'], 90)
        self.assertTrue(propform['low_confidence'])
        self.assertEqual(propform['confidence'], 80)

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
        page = PAGE_PATH.read_text(encoding='utf8')
        script = SCRIPT_PATH.read_text(encoding='utf8')
        self.assertIn('<title>LexNet Explorer</title>', page)
        self.assertIn('explorer.css', page)
        self.assertIn('explorer.js', page)
        self.assertIn('inspector-section', script)
        self.assertIn("document.createElement('ul')", script)
        self.assertIn('lf-values', script)
        self.assertIn('lf-name', script)
        self.assertIn('lf-name-scripts', script)
        self.assertIn('lf-frame', script)
        self.assertIn('lf-constraint', script)
        self.assertIn('lf-merge', script)
        self.assertIn('function appendLexicalName', script)
        self.assertIn('lexical-name-sense', script)
        self.assertIn('lexical-name-scripts', script)
        self.assertIn('example-occurrence', script)
        self.assertIn("atomicLfNames = ['De_nouveau']", script)
        self.assertIn("document.createElement(scriptMarker === '_' ? 'sub' : 'sup')", script)

    def test_local_server_serves_page_and_search_api(self):
        server = create_server(self.queries, '/tmp/example')
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base_url = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base_url + '/') as response:
                page = response.read().decode('utf-8')
            with urlopen(base_url + '/explorer.css') as response:
                stylesheet = response.read().decode('utf-8')
                stylesheet_type = response.headers.get_content_type()
            with urlopen(base_url + '/explorer.js') as response:
                script = response.read().decode('utf-8')
                script_type = response.headers.get_content_type()
            with urlopen(base_url + '/api/search?kind=word&q=chat&mode=exact&forms=1') as response:
                payload = json.load(response)
            with urlopen(base_url + '/api/search?kind=lf&family_id=fam1') as response:
                family_payload = json.load(response)
            with urlopen(base_url + '/api/search?kind=lf&family_id=group%3A2') as response:
                group_payload = json.load(response)
            with urlopen(base_url + '/api/inspect?type=node&id=n3') as response:
                node_payload = json.load(response)
            with urlopen(base_url + '/api/inspect?type=entry&id=e1') as response:
                entry_payload = json.load(response)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertIn('LexNet Explorer', page)
        self.assertIn(':root', stylesheet)
        self.assertEqual(stylesheet_type, 'text/css')
        self.assertIn('function sortBy(key)', script)
        self.assertEqual(script_type, 'text/javascript')
        self.assertEqual({row['node_id'] for row in payload['rows']}, {'n1', 'n2'})
        self.assertEqual(payload['rows'][0]['entry_label']['subscript'], 'N, masc')
        self.assertEqual([row['lf_name'] for row in family_payload['rows']], ['Magn', 'Magn'])
        self.assertEqual([row['lf_name'] for row in group_payload['rows']], ['Oper1'])
        incoming = next(link for link in node_payload['links'] if link.get('direction') == 'Incoming')
        self.assertEqual(incoming['items'][0]['item_id'], 'n1')
        self.assertEqual(entry_payload['title'], 'Lexical entry')


if __name__ == '__main__':
    unittest.main()
