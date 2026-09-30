import json
import threading
import unittest
from urllib.request import urlopen

import pandas as pd

from explorer import CSS_PATH, PAGE_PATH, SCRIPT_PATH, create_server
from explorer_queries import LexnetQueries, split_query


class TestLexnetQueries(unittest.TestCase):
    def setUp(self):
        self.data = {
            'entries': pd.DataFrame([
                {'entry_id': 'e1', 'entry_name': 'chat'},
                {'entry_id': 'e2', 'entry_name': 'prendre le large'},
            ]).set_index('entry_id'),
            'nodes': pd.DataFrame([
                {'node_id': 'n1', 'entry_id': 'e1', 'std_name': 'chat I'},
                {'node_id': 'n2', 'entry_id': 'e1', 'std_name': 'chat II'},
                {'node_id': 'n3', 'entry_id': 'e2', 'std_name': 'prendre le large'},
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
                {'source_node_id': 'n1', 'lexical_function_id': 'lf1', 'target_node_id': 'n2', 'form': '', 'frame': '', 'constraint': ''},
                {'source_node_id': 'n3', 'lexical_function_id': 'lf2', 'target_node_id': 'n1', 'form': 'prendre', 'frame': '', 'constraint': ''},
            ]),
            'definitions': pd.DataFrame(columns=['node_id', 'def_HTML']).set_index('node_id'),
            'labels': pd.DataFrame([
                {'node_id': 'n1', 'semantic_label_id': 'sl1'},
            ]).set_index('node_id'),
            'propforms': pd.DataFrame(columns=['node_id', 'propform']).set_index('node_id'),
            'examples': pd.DataFrame(columns=['example_id', 'content']).set_index('example_id'),
            'ex-rel': pd.DataFrame(columns=['node_id', 'example_id']),
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
        self.assertEqual(set(result.source_name), {'chat I', 'prendre le large'})

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

    def test_lexical_function_links_target_related_units(self):
        outgoing = self.queries.lexical_function_links('n3')
        incoming = self.queries.lexical_function_links('n1')
        self.assertEqual(outgoing[0]['item_id'], 'n1')
        self.assertEqual(outgoing[0]['item_name'], 'chat I')
        self.assertEqual({link['item_id'] for link in incoming}, {'n2', 'n3'})

    def test_entry_inspector_lists_units_with_grammar(self):
        payload = self.queries.inspector_payload('entry', 'e1')
        self.assertEqual(payload['title'], 'Lexical entry')
        self.assertEqual({link['item_id'] for link in payload['links']}, {'n1', 'n2'})
        self.assertIn('chat I  [noun]', payload['description'])

    def test_node_inspector_links_back_to_its_entry(self):
        payload = self.queries.inspector_payload('node', 'n1')
        entry_link = payload['links'][0]
        self.assertEqual(entry_link['item_type'], 'entry')
        self.assertEqual(entry_link['item_id'], 'e1')

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
        self.assertEqual([row['lf_name'] for row in family_payload['rows']], ['Magn'])
        self.assertEqual([row['lf_name'] for row in group_payload['rows']], ['Oper1'])
        self.assertEqual(node_payload['links'][-1]['item_id'], 'n1')
        self.assertEqual(entry_payload['title'], 'Lexical entry')


if __name__ == '__main__':
    unittest.main()
